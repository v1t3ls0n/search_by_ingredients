#!/usr/bin/env python
"""
Build a clean, cached dataset and bulk-index recipes into OpenSearch.

Features:
- Robust logging (console + rotating file logs/indexer.log)
- Reuse cached clean dataset from data/clean/ or data/clean.zip/backup.zip
- Sample mode for fast testing (--sample N)
- Optional skip of image fetching/copying (--skip_images)
- Archive clean product to data/clean.zip (--pack)
- KNN embeddings + alias 'recipes' -> RECIPES_INDEX

Usage:
  python web/index_data.py --force                 # full rebuild + index
  python web/index_data.py --sample 200 --pack     # quick sample + zipped clean
"""

from __future__ import annotations

import os
import re
import string
import logging
import sys
import shutil
import zipfile
import json
from time import sleep
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from decouple import config
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Try to import swifter for parallel processing
try:
    import swifter  # noqa
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False

# ── local helpers (diet classification) ───────────────────────────
from diet_classifiers import is_keto, is_vegan, diet_score  # noqa: E402

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────


def setup_logging(verbosity: int = 1) -> logging.Logger:
    """
    Configure console + rotating file logging.
    logs/indexer.log keeps up to ~5 MB * 3 backups.
    """
    logger = logging.getLogger("indexer")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called twice (e.g., Flask reload)
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbosity > 1 else logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s"))

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(log_dir / "indexer.log",
                             maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))

    # Quiet some noisy libs
    for noisy in ("opensearchpy", "urllib3", "opensearch"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


log = setup_logging()

# ── config & embed model ─────────────────────────────────────────
OPENSEARCH_URL = config(
    "OPENSEARCH_URL", default="http://localhost:9200").strip()
RECIPES_INDEX = config("RECIPES_INDEX",  default="recipes_v2").strip()
EMBED_MODEL = config(
    "EMBED_MODEL", default="sentence-transformers/all-MiniLM-L6-v2")
ENCODE_BATCH_SIZE = config("ENCODE_BATCH_SIZE", default=64, cast=int)

_embed = SentenceTransformer(EMBED_MODEL)


# ─────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, set)):
        return "\n".join(str(i) for i in x)
    if isinstance(x, np.ndarray):
        return "\n".join(str(i) for i in x.tolist())
    return str(x)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        return [p.strip() for p in re.split(r"[\n,]+", x) if p.strip()]
    return [str(x)]


def make_uid(row: dict, data_source: str) -> str:
    """
    Stable unique ID (independent of row position).
    Uses source + title + ingredients + raw id.
    """
    title = row.get("title") or row.get("Title") or ""
    ings = row.get("ingredients") or row.get(
        "Ingredients") or row.get("Cleaned_Ingredients") or ""
    rawid = row.get("id") or ""
    base = f"{data_source}|{rawid}|{title}|{ings}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

# ─────────────────────────────────────────────────────────────────
# Image fetching / pooling
# ─────────────────────────────────────────────────────────────────


def _wayback_resolve(original_url: str, ts="20170320000000") -> str | None:
    """Resolve a wildcard Wayback URL to a specific snapshot if possible."""
    try:
        orig = original_url
        if "web.archive.org" in original_url and "*/" in original_url:
            orig = original_url.split("*/", 1)[1]
        r = requests.get(
            "https://archive.org/wayback/available",
            params={"url": orig, "timestamp": ts},
            timeout=10
        )
        j = r.json()
        closest = j.get("archived_snapshots", {}).get("closest")
        if closest and closest.get("available") and closest.get("url"):
            return closest["url"]
    except Exception as e:
        log.debug("Wayback resolve failed for %s: %s", original_url, e)
    return None


def fetch_allrecipes_images(df: pd.DataFrame, out_dir="data/images", max_workers: int = 16) -> dict[str, list[str]]:
    """
    Download Allrecipes images (via Wayback) into a unified pool.
    Returns: { id: [local_paths...] }  (usually one image per recipe)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def task(row) -> Tuple[str, list[str]]:
        url = row.get("photo_url")
        rid = row.get("id")
        if not rid or not isinstance(url, str) or not url:
            return (rid, [])

        fetch_url = url
        if "*/" in url:
            resolved = _wayback_resolve(url)
            if resolved:
                fetch_url = resolved

        dest = out / f"{rid}_1.jpg"
        if dest.exists():
            return (rid, [str(dest)])

        try:
            r = requests.get(fetch_url, timeout=15)
            if r.status_code == 200 and r.content:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return (rid, [str(dest)])
            else:
                log.debug("Allrecipes fetch failed %s → %s",
                          fetch_url, r.status_code)
        except Exception as e:
            log.debug("Allrecipes fetch error %s: %s", fetch_url, e)
        return (rid, [])

    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(task, row) for _, row in df.iterrows()
                if isinstance(row.get("photo_url"), str)]
        for fut in as_completed(futs):
            rid, paths = fut.result()
            if rid:
                results[rid] = paths
    return results


def pool_kaggle_images(df: pd.DataFrame, out_dir="data/images") -> dict[str, list[str]]:
    """
    Copy Kaggle images (Image_Name.*) into the unified pool as {id}_{n}.*.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_dir = Path(
        "data/food-ingredients-and-recipe-dataset-with-image/Food Images")

    def matches_for_row(row) -> Tuple[str, list[str]]:
        rid = row.get("id")
        if not rid:
            return (None, [])

        stem = None
        if isinstance(row.get("Image_Name"), str):
            stem = row["Image_Name"]
        else:
            p = row.get("photo_url")
            if isinstance(p, str):
                stem = Path(p).stem

        if not stem:
            return (rid, [])

        files = sorted(base_dir.glob(stem + ".*"))
        pooled: list[str] = []
        for idx, src in enumerate(files, 1):
            ext = src.suffix.lower()
            dest = out / f"{rid}_{idx}{ext}"
            if not dest.exists():
                try:
                    shutil.copy(src, dest)
                except Exception as e:
                    log.debug("Copy failed %s → %s: %s", src, dest, e)
                    continue
            pooled.append(str(dest))
        return (rid, pooled)

    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(matches_for_row, row) for _, row in df.iterrows()]
        for fut in as_completed(futs):
            rid, paths = fut.result()
            if rid:
                results[rid] = paths
    return results

# ─────────────────────────────────────────────────────────────────
# Loading / normalizing tables
# ─────────────────────────────────────────────────────────────────


def load_table_any(path: Path) -> pd.DataFrame:
    """
    Load recipes from Parquet/CSV/JSON by extension.
    Normalizes dataset schema and adds a data_source column
    based on the file path.
    """
    expected_cols = ["id", "title", "description",
                     "ingredients", "instructions", "photo_url", "data_source"]

    if not path.exists():
        log.warning("Data file not found: %s", path)
        return pd.DataFrame(columns=expected_cols)

    suffix = path.suffix.lower()
    df: pd.DataFrame | None = None

    try:
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".json":
            try:
                df = pd.read_json(path, lines=True)
            except ValueError:
                df = pd.read_json(path)
        else:
            try:
                df = pd.read_parquet(path)
            except Exception:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_json(path, lines=True)
    except Exception as e:
        log.warning(
            "Failed to load %s (%s). Proceeding with empty dataset.", path, e)
        return pd.DataFrame(columns=expected_cols)

    if df is None or df.empty:
        return pd.DataFrame(columns=expected_cols)

    data_source = f"data/{path.parent.name}_{path.stem}{path.suffix}"

    # Kaggle-case
    if "Image_Name" in df.columns:
        base_dir = Path(
            "data/food-ingredients-and-recipe-dataset-with-image/Food Images")

        def build_path(name: str):
            if not isinstance(name, str):
                return None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = base_dir / f"{name}{ext}"
                if candidate.exists():
                    return str(candidate)
            return None

        df = pd.DataFrame({
            "id": df.apply(lambda r: make_uid(r, data_source), axis=1),
            "title": df["Title"],
            "description": None,
            "ingredients": df["Cleaned_Ingredients"].fillna(df["Ingredients"]),
            "instructions": df["Instructions"],
            "photo_url": df["Image_Name"].apply(build_path),
            "data_source": data_source
        })

    # Allrecipes / others
    else:
        if "id" not in df.columns:
            df["id"] = df.apply(lambda r: make_uid(r, data_source), axis=1)
        if "photo_url" not in df.columns:
            df["photo_url"] = None

        def rewrite_photo_url(url):
            if not isinstance(url, str) or not url.strip():
                return None
            low = url.lower()
            if "nophoto" in low or "nopic" in low:
                return None
            return f"https://web.archive.org/web/20170320000000*/{url}"

        df["photo_url"] = df["photo_url"].apply(rewrite_photo_url)
        df["data_source"] = data_source

    return df[expected_cols]

# ─────────────────────────────────────────────────────────────────
# OpenSearch helpers
# ─────────────────────────────────────────────────────────────────


def wait_for_os(client: OpenSearch, retries: int = 30, delay: int = 2) -> bool:
    for _ in range(retries):
        if client.ping():
            return True
        sleep(delay)
    return False


def delete_indices(client: OpenSearch) -> None:
    for idx in (RECIPES_INDEX, "recipes", "ingredients"):
        if client.indices.exists(index=idx):
            client.indices.delete(index=idx)
            log.info("Deleted index %s", idx)


def ensure_recipes_knn_index(client: OpenSearch) -> None:
    """
    Ensure index exists with KNN + image_files, and alias 'recipes' -> RECIPES_INDEX.
    """
    body = {
        "settings": {"index": {"knn": True, "refresh_interval": "1s"}},
        "mappings": {"properties": {
            "id":           {"type": "keyword"},
            "title":        {"type": "text"},
            "description":  {"type": "text"},
            "ingredients":  {"type": "text"},
            "instructions": {"type": "text"},
            "photo_url":    {"type": "keyword"},
            "image_files":  {"type": "keyword"},
            "keto":         {"type": "boolean"},
            "vegan":        {"type": "boolean"},
            "keto_score":   {"type": "integer"},
            "vegan_score":  {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {"name": "hnsw", "space_type": "cosinesimil"}
            },
        }},
    }

    if not client.indices.exists(index=RECIPES_INDEX):
        client.indices.create(index=RECIPES_INDEX, body=body)
        log.info("Created index %s (KNN enabled)", RECIPES_INDEX)
    else:
        mapping = client.indices.get_mapping(index=RECIPES_INDEX)
        props = mapping.get(RECIPES_INDEX, {}).get(
            "mappings", {}).get("properties", {}) or {}

        missing = {}
        if props.get("embedding", {}).get("type") != "knn_vector":
            missing["embedding"] = {
                "type": "knn_vector",
                "dimension": 384,
                "method": {"name": "hnsw", "space_type": "cosinesimil"},
            }
        if "image_files" not in props:
            missing["image_files"] = {"type": "keyword"}

        if missing:
            log.info("Upgrading %s mapping to add: %s",
                     RECIPES_INDEX, ", ".join(missing.keys()))
            client.indices.close(index=RECIPES_INDEX)
            client.indices.put_settings(index=RECIPES_INDEX, body={
                                        "index": {"knn": True}})
            client.indices.put_mapping(index=RECIPES_INDEX, body={
                                       "properties": missing})
            client.indices.open(index=RECIPES_INDEX)
            log.info("Mapping upgraded for %s.", RECIPES_INDEX)

    # ensure alias
    try:
        alias_lookup = client.indices.get_alias(name="recipes", ignore=[404])
        actions = []
        if isinstance(alias_lookup, dict) and alias_lookup and alias_lookup.get("status") != 404:
            current_targets = list(alias_lookup.keys())
            for idx in current_targets:
                if idx != RECIPES_INDEX:
                    actions.append(
                        {"remove": {"index": idx, "alias": "recipes"}})
            if RECIPES_INDEX not in current_targets:
                actions.append(
                    {"add": {"index": RECIPES_INDEX, "alias": "recipes"}})
            if actions:
                client.indices.update_aliases({"actions": actions})
                log.info("Alias 'recipes' moved to -> %s", RECIPES_INDEX)
        else:
            client.indices.put_alias(index=RECIPES_INDEX, name="recipes")
            log.info("Alias 'recipes' -> %s created", RECIPES_INDEX)
    except Exception as e:
        log.warning(
            "Alias management warning: %s; ensuring simple alias add (%s)", RECIPES_INDEX, e)
        client.indices.put_alias(index=RECIPES_INDEX, name="recipes")


def ensure_alias(client: OpenSearch, alias: str, index: str) -> None:
    try:
        if client.indices.exists_alias(name=alias):
            targets = list(client.indices.get_alias(name=alias).keys())
            if targets != [index]:
                for t in targets:
                    client.indices.delete_alias(index=t, name=alias)
                client.indices.put_alias(index=index, name=alias)
        else:
            client.indices.put_alias(index=index, name=alias)
        log.info("Alias %s → %s ensured", alias, index)
    except Exception as e:
        log.warning("ensure_alias(%s→%s) failed: %s", alias, index, e)


def ensure_ingredients_index(client: OpenSearch) -> None:
    if not client.indices.exists(index="ingredients"):
        client.indices.create(index="ingredients",
                              body={"mappings": {"properties": {"ingredients": {"type": "text"}}}})
        log.info("Created index ingredients")

# ─────────────────────────────────────────────────────────────────
# Diet enrichment
# ─────────────────────────────────────────────────────────────────


def normalize_ingredient(txt: str) -> str:
    """Loose normaliser so '2 cups strawberries' → 'strawberry'."""
    if not isinstance(txt, str):
        return str(txt)
    txt = txt.lower().strip()
    txt = txt.rsplit(",", 1)[0]
    txt = re.sub(r"\([^()]+\)", "", txt)
    txt = re.sub(r"\d+\s*\d*/\d*", "", txt)
    txt = txt.translate({ord(c): " " for c in string.punctuation})
    units = ("cup cups can cans tablespoon tablespoons tbsp teaspoon teaspoons tsp ounce ounces oz "
             "pound pounds lb lbs gram grams g kilogram kilograms kg milliliter milliliters ml "
             "liter liters l pinch pinches dash dashes piece pieces slice slices small medium large "
             "cube cubes inch inches cm mm quart quarts qt jar scoop scoops gallon gallons gal pint "
             "pints pt fluid ounce fluid ounces fl oz package packages pkg pack packs").split()
    txt = re.sub(rf"\b({'|'.join(units)})\b", "", txt)
    if txt.endswith("ies"):
        txt = txt[:-3]+"y"
    elif txt.endswith("es") and txt[-3] in "sxz":
        txt = txt[:-2]
    elif txt.endswith("s"):
        txt = txt[:-1]
    return " ".join(txt.split())


def enrich_recipes_df(df: pd.DataFrame) -> pd.DataFrame:
    if "ingredients" not in df.columns:
        df["ingredients"] = [[] for _ in range(len(df))]
    else:
        df["ingredients"] = df["ingredients"].apply(_as_list)

    ing_series = df["ingredients"]

    if SWIFTER_AVAILABLE:
        log.info("Applying diet classifiers with swifter (parallel)…")
        apply = ing_series.swifter.apply
    else:
        log.info("Swifter not available; using standard pandas")
        apply = ing_series.apply

    df["keto"] = apply(is_keto)
    df["vegan"] = apply(is_vegan)
    df["keto_score"] = apply(lambda ings: diet_score(ings, "keto"))
    df["vegan_score"] = apply(lambda ings: diet_score(ings, "vegan"))
    return df

# ─────────────────────────────────────────────────────────────────
# Embeddings + bulk index
# ─────────────────────────────────────────────────────────────────


def _doc_text_for_embedding(doc: dict) -> str:
    t = _to_text(doc.get("title"))
    ing = _to_text(doc.get("ingredients"))
    instr = doc.get("instructions")
    if instr is None:
        instr = doc.get("directions")
    ins = _to_text(instr)
    desc = _to_text(doc.get("description"))
    return f"{t}\nIngredients:\n{ing}\nInstructions:\n{ins}\n{desc}"


def build_doc_embedding(doc: dict) -> list[float]:
    v = _embed.encode([_doc_text_for_embedding(doc)],
                      convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.tolist()


def bulk_index_recipes(client: OpenSearch, records: List[Dict], batch_size: int = 1024):
    df = enrich_recipes_df(pd.DataFrame(records))
    enriched: List[Dict] = df.to_dict("records")

    texts = [_doc_text_for_embedding(r) for r in enriched]
    vecs: List[List[float]] = []
    for j in range(0, len(texts), ENCODE_BATCH_SIZE):
        chunk = texts[j:j + ENCODE_BATCH_SIZE]
        m = _embed.encode(
            chunk,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs.extend(v.tolist() for v in m)
    for r, v in zip(enriched, vecs):
        r["embedding"] = v

    actions: List[Dict] = []
    vocab: Set[str] = set()

    for r in enriched:
        _id = str(r.get("id") or r.get("title") or os.urandom(8).hex())
        actions.append({"index": {"_index": RECIPES_INDEX, "_id": _id}})
        actions.append(r)

        for i in _as_list(r.get("ingredients")):
            token = normalize_ingredient(i)
            if token:
                vocab.add(token)

        if len(actions) >= batch_size * 2:
            client.bulk(actions)
            actions.clear()

    if actions:
        client.bulk(actions)

    if vocab:
        ing_actions: List[Dict] = []
        for token in vocab:
            ing_actions += [{"index": {"_index": "ingredients"}},
                            {"ingredients": token}]
        client.bulk(ing_actions)

# ─────────────────────────────────────────────────────────────────
# Clean product (cache) builder
# ─────────────────────────────────────────────────────────────────


def _unzip_if_exists(zip_path: Path, dest_dir: Path) -> bool:
    if not zip_path.exists():
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Unzipping %s → %s …", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return True


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    log.info("Creating archive %s from %s …", zip_path, src_dir)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            zf.write(p, p.relative_to(src_dir))


def load_or_build_clean_dataset(
    base_allrecipes: Path,
    kaggle_csv: Path,
    sample: int = 0,
    skip_images: bool = False,
    pack: bool = False,
) -> pd.DataFrame:
    """
    Reuse clean product if present, else build it.
    Clean product layout:
      data/clean/
        ├─ recipes.parquet
        └─ images/  (pooled images {id}_{i}.ext)

    Return: DataFrame loaded from clean product.
    """
    clean_dir = Path("data/clean")
    clean_zip = Path("data/clean.zip")
    backup_zip = Path("data/backup.zip")

    # 1) If clean dir exists, reuse
    if clean_dir.exists() and (clean_dir / "recipes.parquet").exists():
        log.info("Found existing clean dataset at %s — reusing.", clean_dir)
        return pd.read_parquet(clean_dir / "recipes.parquet")

    # 2) Else unzip a packaged clean dataset if available
    if _unzip_if_exists(clean_zip, clean_dir) or _unzip_if_exists(backup_zip, clean_dir):
        if (clean_dir / "recipes.parquet").exists():
            log.info("Loaded clean dataset from zip.")
            return pd.read_parquet(clean_dir / "recipes.parquet")
        else:
            log.warning(
                "Zip extracted but recipes.parquet missing — will rebuild.")

    # 3) Build from raw sources
    log.info("Building clean dataset from raw sources …")
    df_all = load_table_any(
        base_allrecipes) if base_allrecipes.exists() else pd.DataFrame()
    if not df_all.empty:
        # ensure IDs if needed (should already be there)
        if "id" not in df_all.columns or df_all["id"].isna().any():
            df_all["id"] = df_all.apply(lambda r: make_uid(r.to_dict(
            ), f"data/{base_allrecipes.parent.name}_{base_allrecipes.stem}{base_allrecipes.suffix}"), axis=1)

    if kaggle_csv.exists():
        df_kag = load_table_any(kaggle_csv)

        # attach raw Image_Name for multi-image glob via a consistent data_source
        try:
            raw_df = pd.read_csv(
                kaggle_csv,
                usecols=["Image_Name", "Title", "Ingredients",
                         "Instructions", "Cleaned_Ingredients"]
            )
            src_ds = f"data/{kaggle_csv.parent.name}_{kaggle_csv.stem}{kaggle_csv.suffix}"
            tmp = pd.DataFrame({
                "Image_Name": raw_df["Image_Name"],
                "id": raw_df.apply(lambda r: make_uid({
                    "Title": r.get("Title"),
                    "Ingredients": r.get("Ingredients"),
                    "Cleaned_Ingredients": r.get("Cleaned_Ingredients"),
                }, src_ds), axis=1)
            })
            df_kag = df_kag.merge(tmp, on="id", how="left")
        except Exception:
            pass
    else:
        df_kag = pd.DataFrame()

    # Sample (fast path) — do this BEFORE heavy image work
    if sample and sample > 0:
        if not df_all.empty:
            df_all = df_all.sample(
                n=min(sample, len(df_all)), random_state=42).reset_index(drop=True)
        if not df_kag.empty:
            df_kag = df_kag.sample(
                n=min(sample, len(df_kag)), random_state=42).reset_index(drop=True)
        log.info("Sample mode: using up to %d rows from each dataset.", sample)

    # Build pooled images into clean_dir/images/
    images_dir = clean_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not skip_images:
        if not df_all.empty:
            log.info("Fetching Allrecipes images via Wayback (parallel)…")
            all_map = fetch_allrecipes_images(
                df_all, out_dir=str(images_dir), max_workers=16)
            df_all["image_files"] = df_all["id"].map(
                all_map).apply(lambda x: x or [])
            df_all["photo_url"] = df_all["image_files"].apply(
                lambda xs: xs[0] if xs else None)

        if not df_kag.empty:
            log.info("Pooling Kaggle images (copying local files) …")
            kag_map = pool_kaggle_images(df_kag, out_dir=str(images_dir))
            df_kag["image_files"] = df_kag["id"].map(
                kag_map).apply(lambda x: x or [])
            df_kag["photo_url"] = df_kag.apply(
                lambda r: (r["image_files"][0] if r.get(
                    "image_files") else r.get("photo_url")),
                axis=1
            )
    else:
        # Ensure columns exist even if skipping
        for df in (df_all, df_kag):
            if not df.empty:
                if "image_files" not in df.columns:
                    df["image_files"] = [[] for _ in range(len(df))]

    # Merge
    df_all = df_all if not df_all.empty else pd.DataFrame(
        columns=["id", "title", "description", "ingredients", "instructions", "photo_url", "image_files", "data_source"])
    df_kag = df_kag if not df_kag.empty else pd.DataFrame(
        columns=["id", "title", "description", "ingredients", "instructions", "photo_url", "image_files", "data_source"])
    df = pd.concat([df_all, df_kag], ignore_index=True)

    # Save clean product
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_pq = clean_dir / "recipes.parquet"
    df.to_parquet(out_pq, index=False)
    log.info("Saved clean dataset: %s (%d rows)", out_pq, len(df))

    if pack:
        _zip_dir(clean_dir, clean_zip)
        log.info("Packed clean dataset → %s", clean_zip)

    return df

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────


def main(cfg):
    # Reconfigure log level by verbosity
    if cfg.verbose:
        setup_logging(verbosity=2)

    client = OpenSearch(
        hosts=[config("OPENSEARCH_URL", default=OPENSEARCH_URL)],
        use_ssl=False, verify_certs=False, ssl_show_warn=False,
    )
    if not wait_for_os(client):
        log.error("OpenSearch not reachable at %s", config(
            "OPENSEARCH_URL", default=OPENSEARCH_URL))
        sys.exit(1)

    if cfg.force:
        delete_indices(client)

    ensure_recipes_knn_index(client)
    ensure_ingredients_index(client)

    # Clean product path decisions
    base_allrecipes = Path(cfg.data_file)
    kaggle_csv = Path(
        "data/food-ingredients-and-recipe-dataset-with-image/dataset.csv")

    # Reuse / build clean product
    df = load_or_build_clean_dataset(
        base_allrecipes=base_allrecipes,
        kaggle_csv=kaggle_csv,
        sample=cfg.sample,
        skip_images=cfg.skip_images,
        pack=cfg.pack,
    )

    records = df.to_dict("records") if not df.empty else []
    with tqdm(total=len(records), desc=f"Indexing into {RECIPES_INDEX}") as bar:
        for i in range(0, len(records), cfg.batch_size):
            bulk_index_recipes(
                client, records[i:i + cfg.batch_size], batch_size=cfg.batch_size)
            bar.update(min(cfg.batch_size, len(records) - i))

    ensure_alias(client, "recipes", RECIPES_INDEX)
    log.info("✔ Done – %s recipes indexed into %s",
             len(records), RECIPES_INDEX)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Bulk size per request.")
    p.add_argument("--data_file", type=Path,
                   default=Path("data/allrecipes.parquet"))
    p.add_argument("--force", action="store_true",
                   help="Delete & re-index OpenSearch indices.")
    p.add_argument("--sample", type=int, default=0,
                   help="Process only N rows from each dataset (0 = all).")
    p.add_argument("--skip_images", action="store_true",
                   help="Skip downloading/copying images (faster).")
    p.add_argument("--pack", action="store_true",
                   help="Zip data/clean/ to data/clean.zip after building.")
    p.add_argument("--verbose", action="store_true",
                   help="More verbose console logging.")
    main(p.parse_args())
