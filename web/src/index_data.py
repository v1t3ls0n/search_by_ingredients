#!/usr/bin/env python
"""
Bulk-index Allrecipes data into OpenSearch with diet metadata + KNN embeddings.

Usage (inside container)
------------------------
python web/index_data.py --force
"""
from __future__ import annotations

import os
import re
import string
import logging
import sys
from time import sleep
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from decouple import config
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib
# Try to import swifter for parallel processing
try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False

# ── local helpers (diet classification) ───────────────────────────
from diet_classifiers import is_keto, is_vegan, diet_score  # noqa: E402

import numpy as np  # add this near the top with other imports

from pathlib import Path

import hashlib
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── logging ───────────────────────────────────────────────────────
logging.getLogger("opensearch").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── config & embed model ─────────────────────────────────────────
OPENSEARCH_URL = config("OPENSEARCH_URL", default="http://localhost:9200")
RECIPES_INDEX = config("RECIPES_INDEX",  default="recipes_v2")
EMBED_MODEL = config(
    "EMBED_MODEL",    default="sentence-transformers/all-MiniLM-L6-v2")
ENCODE_BATCH_SIZE = config("ENCODE_BATCH_SIZE", default=64, cast=int)

_embed = SentenceTransformer(EMBED_MODEL)


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


def fetch_allrecipes_images(df: pd.DataFrame, out_dir="data/images/allrecipes"):
    """
    Download all Allrecipes images (via web.archive.org) into a local directory.
    Uses the unique id as filename. Skips if file already exists.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        url = row.get("photo_url")
        rid = row.get("id")
        if not url or not isinstance(url, str):
            continue
        dest = out / f"{rid}.jpg"
        if dest.exists():
            continue

        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200 and r.content:
                with open(dest, "wb") as f:
                    f.write(r.content)
                log.info("Saved image %s", dest)
            else:
                log.warning("Failed %s → %s", url, r.status_code)
        except Exception as e:
            log.warning("Error fetching %s: %s", url, e)


def make_uid(row: dict, data_source: str) -> str:
    """
    Stable unique ID (independent of row position).
    Uses source + title + ingredients (falling back to other fields if needed).
    """
    title = row.get("title") or row.get("Title") or ""
    ings = row.get("ingredients") or row.get(
        "Ingredients") or row.get("Cleaned_Ingredients") or ""
    rawid = row.get("id") or ""
    base = f"{data_source}|{rawid}|{title}|{ings}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _wayback_resolve(original_url: str, ts="20170320000000") -> str | None:
    """
    Use the Wayback 'available' API to get a concrete snapshot URL for original_url.
    Returns None if not found.
    """
    try:
        # If we stored wildcard URLs, strip the '*/' part back to original.
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

    def task(row) -> tuple[str, list[str]]:
        url = row.get("photo_url")
        rid = row.get("id")
        if not rid or not isinstance(url, str) or not url:
            return (rid, [])

        # One image per recipe (index 1); keep space for future multi-snapshots if needed
        # Try to resolve wildcard to concrete snapshot
        fetch_url = url
        if "*/" in url:
            resolved = _wayback_resolve(url)
            if resolved:
                fetch_url = resolved

        # choose jpg as container; we don't know original format reliably
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
    Copy all matching Kaggle images (Image_Name.*) into the unified pool as {id}_{n}.*.
    Assumes df has columns: id, and either:
      - 'Image_Name' (raw CSV) OR
      - 'photo_url' pointing to one of the image files whose stem is the image name.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # base dir where Kaggle images live
    base_dir = Path(
        "data/food-ingredients-and-recipe-dataset-with-image/Food Images")

    def matches_for_row(row) -> tuple[str, list[str]]:
        rid = row.get("id")
        if not rid:
            return (None, [])

        # Figure out the stem to glob
        stem = None
        if isinstance(row.get("Image_Name"), str):
            stem = row["Image_Name"]
        else:
            p = row.get("photo_url")
            if isinstance(p, str):
                stem = Path(p).stem

        if not stem:
            return (rid, [])

        # Find all files with that stem
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

    # unique data_source string derived from file path
    data_source = f"data/{path.parent.name}_{path.stem}{path.suffix}"

    # ── Case 1: Kaggle food dataset with Image_Name ───────────
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
            "description": None,  # dataset has no description field
            "ingredients": df["Cleaned_Ingredients"].fillna(df["Ingredients"]),
            "instructions": df["Instructions"],
            "photo_url": df["Image_Name"].apply(build_path),
            "data_source": data_source
        })

    # ── Case 2: Allrecipes / other ───────────
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


def _doc_text_for_embedding(doc: dict) -> str:
    t = _to_text(doc.get("title"))
    ing = _to_text(doc.get("ingredients"))

    instr = doc.get("instructions")
    if instr is None:                       # ← DO THIS, don't use “or” on arrays
        instr = doc.get("directions")
    ins = _to_text(instr)

    desc = _to_text(doc.get("description"))
    return f"{t}\nIngredients:\n{ing}\nInstructions:\n{ins}\n{desc}"


def build_doc_embedding(doc: dict) -> list[float]:
    v = _embed.encode([_doc_text_for_embedding(doc)],
                      convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.tolist()

# ── OpenSearch helpers ────────────────────────────────────────────


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
    Ensure the target recipes index exists with a KNN embedding field,
    and ensure alias 'recipes' points to it for backward compatibility.
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
            # ← NEW (array of local paths)
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

    # 1) Create index if missing
    if not client.indices.exists(index=RECIPES_INDEX):
        client.indices.create(index=RECIPES_INDEX, body=body)
        log.info("Created index %s (KNN enabled)", RECIPES_INDEX)
    else:
        # 2) Ensure mapping has knn_vector 'embedding' (upgrade in-place if needed)
        mapping = client.indices.get_mapping(index=RECIPES_INDEX)
        props = mapping.get(RECIPES_INDEX, {}).get(
            "mappings", {}).get("properties", {}) or {}
        needs_embedding = props.get(
            "embedding", {}).get("type") != "knn_vector"

        if needs_embedding:
            log.info(
                "Upgrading %s mapping to add knn_vector embedding…", RECIPES_INDEX)
            # make sure knn is enabled
            client.indices.close(index=RECIPES_INDEX)
            client.indices.put_settings(index=RECIPES_INDEX, body={
                                        "index": {"knn": True}})
            client.indices.put_mapping(
                index=RECIPES_INDEX,
                body={"properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {"name": "hnsw", "space_type": "cosinesimil"},
                    }
                }},
            )
            client.indices.open(index=RECIPES_INDEX)
            log.info("Mapping upgraded for %s.", RECIPES_INDEX)

    # 3) Ensure alias 'recipes' -> RECIPES_INDEX
    #    (so your Flask app that queries index='recipes' keeps working)
    try:
        # Does alias exist and where does it point?
        alias_lookup = client.indices.get_alias(name="recipes", ignore=[404])
        actions = []

        if isinstance(alias_lookup, dict) and alias_lookup and alias_lookup.get("status") != 404:
            current_targets = list(alias_lookup.keys())
            for idx in current_targets:
                if idx != RECIPES_INDEX:
                    actions.append(
                        {"remove": {"index": idx, "alias": "recipes"}})
            # ensure it's added to the desired index
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
        # fallback: try simple put_alias
        log.warning(
            "Alias management warning: %s; ensuring simple alias add", e)
        client.indices.put_alias(index=RECIPES_INDEX, name="recipes")
        log.info("Alias 'recipes' -> %s ensured", RECIPES_INDEX)


def ensure_alias(client: OpenSearch, alias: str, index: str) -> None:
    """Point alias -> index (and detach from any other indices)."""
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

# ── diet enrichment & vocab ---------------------------------------


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
        log.warning("Swifter not available; using standard pandas")
        apply = ing_series.apply

    df["keto"] = apply(is_keto)
    df["vegan"] = apply(is_vegan)
    df["keto_score"] = apply(lambda ings: diet_score(ings, "keto"))
    df["vegan_score"] = apply(lambda ings: diet_score(ings, "vegan"))
    return df

# ── bulk index with embeddings -----------------------------------


def bulk_index_recipes(client: OpenSearch, records: List[Dict], batch_size: int = 1024):
    # Diet flags first (keeps your swifter path)
    df = enrich_recipes_df(pd.DataFrame(records))
    enriched: List[Dict] = df.to_dict("records")

    # ----- NEW: embed in mini-batches, not per row -----
    texts = [_doc_text_for_embedding(r) for r in enriched]
    vecs: List[List[float]] = []
    for j in range(0, len(texts), ENCODE_BATCH_SIZE):
        chunk = texts[j:j + ENCODE_BATCH_SIZE]
        # Turn off the internal progress bar; we drive our own logging
        m = _embed.encode(
            chunk,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs.extend(v.tolist() for v in m)
    # attach embeddings back
    for r, v in zip(enriched, vecs):
        r["embedding"] = v

    # ----- Bulk index + vocab build -----
    actions: List[Dict] = []
    vocab: Set[str] = set()

    for r in enriched:
        _id = str(r.get("id") or r.get("title") or os.urandom(8).hex())
        actions.append({"index": {"_index": RECIPES_INDEX, "_id": _id}})
        actions.append(r)

        # robust ingredient tokenization for vocab
        for i in _as_list(r.get("ingredients")):
            token = normalize_ingredient(i)
            if token:
                vocab.add(token)

        if len(actions) >= batch_size * 2:  # 2 lines per doc in bulk API
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


# ── main ----------------------------------------------------------


def main(cfg):
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

    # ---------- Load Allrecipes ----------
    df_all = load_table_any(Path(cfg.data_file))
    if not df_all.empty:
        # data_source already set by load_table_any; ensure IDs exist
        if "id" not in df_all.columns or df_all["id"].isna().any():
            df_all["id"] = df_all.apply(lambda r: make_uid(
                r.to_dict(), str(cfg.data_file)), axis=1)

    # ---------- Load Kaggle CSV ----------
    kaggle_csv = Path(
        "data/food-ingredients-and-recipe-dataset-with-image/dataset.csv")
    if kaggle_csv.exists():
        # Use load_table_any to normalize text fields; it will set photo_url to one path if present
        df_kag = load_table_any(kaggle_csv)
        # Ensure we have IDs (based on normalized fields + data_source)
        if "id" not in df_kag.columns or df_kag["id"].isna().any():
            df_kag["id"] = df_kag.apply(lambda r: make_uid(
                r.to_dict(), str(kaggle_csv)), axis=1)
        # Try to keep raw Image_Name if present (for multi-image glob), else we’ll derive from photo_url
        try:
            raw_df = pd.read_csv(kaggle_csv, usecols=[
                                 "Image_Name", "Title", "Ingredients", "Instructions", "Cleaned_Ingredients"])
            # align by stable uid
            tmp = pd.DataFrame({
                "Image_Name": raw_df["Image_Name"],
                # recompute uid the same way (source: kaggle_csv)
                "id": raw_df.apply(lambda r: make_uid({
                    "Title": r.get("Title"),
                    "Ingredients": r.get("Ingredients"),
                    "Cleaned_Ingredients": r.get("Cleaned_Ingredients"),
                }, str(kaggle_csv)), axis=1)
            })
            df_kag = df_kag.merge(tmp, on="id", how="left")
        except Exception:
            # not fatal
            pass
    else:
        df_kag = pd.DataFrame(columns=[
                              "id", "title", "description", "ingredients", "instructions", "photo_url", "data_source"])

    # ---------- Build unified image pool ----------
    pool_dir = Path("data/images")
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Allrecipes: download via Wayback (parallel)
    all_map = fetch_allrecipes_images(df_all, out_dir=str(
        pool_dir), max_workers=16) if not df_all.empty else {}

    # Kaggle: copy all matching images (Image_Name.*) into pool
    kag_map = pool_kaggle_images(df_kag, out_dir=str(
        pool_dir)) if not df_kag.empty else {}

    # Attach 'image_files' + representative 'photo_url' from pool (first image)
    if not df_all.empty:
        df_all["image_files"] = df_all["id"].map(
            all_map).apply(lambda x: x or [])
        df_all["photo_url"] = df_all["image_files"].apply(
            lambda xs: xs[0] if xs else None)

    if not df_kag.empty:
        df_kag["image_files"] = df_kag["id"].map(
            kag_map).apply(lambda x: x or [])
        # prefer a pooled path; if none, keep whatever load_table_any gave
        df_kag["photo_url"] = df_kag.apply(
            lambda r: (r["image_files"][0] if r.get("image_files") else r.get("photo_url")), axis=1
        )

    # ---------- Combine & index ----------
    df_all = df_all if not df_all.empty else pd.DataFrame(
        columns=["id", "title", "description", "ingredients", "instructions", "photo_url", "image_files", "data_source"])
    df_kag = df_kag if not df_kag.empty else pd.DataFrame(
        columns=["id", "title", "description", "ingredients", "instructions", "photo_url", "image_files", "data_source"])
    df = pd.concat([df_all, df_kag], ignore_index=True)

    if df.empty:
        log.warning(
            "No records loaded. Indexes will exist but contain no recipes.")
        records = []
    else:
        # important: keep image_files for indexing (keyword array)
        records = df.to_dict("records")

    with tqdm(total=len(records), desc=f"Indexing into {RECIPES_INDEX}") as bar:
        for i in range(0, len(records), cfg.batch_size):
            bulk_index_recipes(
                client, records[i:i + cfg.batch_size], batch_size=cfg.batch_size)
            bar.update(min(cfg.batch_size, len(records) - i))

    # Point a friendly alias "recipes" to the current physical index
    ensure_alias(client, "recipes", RECIPES_INDEX)

    log.info("✔ Done – %s recipes indexed into %s",
             len(records), RECIPES_INDEX)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--data_file", type=Path,
                   default=Path("data/allrecipes.parquet"))

    p.add_argument("--force", action="store_true",
                   help="Delete & re-index data")
    main(p.parse_args())
