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

# Try to import swifter for parallel processing
try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False

# ── local helpers (diet classification) ───────────────────────────
from diet_classifiers import is_keto, is_vegan, diet_score  # noqa: E402

import numpy as np  # add this near the top with other imports


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


# ── logging ───────────────────────────────────────────────────────
logging.getLogger("opensearch").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── config & embed model ─────────────────────────────────────────
OPENSEARCH_URL = config("OPENSEARCH_URL", default="http://localhost:9200")
RECIPES_INDEX = config("RECIPES_INDEX",  default="recipes_v2")
EMBED_MODEL = config(
    "EMBED_MODEL",    default="sentence-transformers/all-MiniLM-L6-v2")

_embed = SentenceTransformer(EMBED_MODEL)


def load_table_any(path: Path) -> pd.DataFrame:
    """
    Load recipes from Parquet/CSV/JSON by extension.
    If the file doesn't exist or is invalid, return an empty DataFrame with expected columns.
    """
    expected_cols = ["id", "title", "description",
                     "ingredients", "instructions", "photo_url"]
    if not path.exists():
        log.warning("Data file not found: %s", path)
        return pd.DataFrame(columns=expected_cols)

    suffix = path.suffix.lower()
    try:
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".csv"}:
            return pd.read_csv(path)
        if suffix in {".json"}:
            # supports both records and line-delimited json
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                return pd.read_json(path)
        # Unknown: try parquet then csv then json
        try:
            return pd.read_parquet(path)
        except Exception:
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_json(path, lines=True)
    except Exception as e:
        log.warning(
            "Failed to load %s (%s). Proceeding with empty dataset.", path, e)
        return pd.DataFrame(columns=expected_cols)


def _doc_text_for_embedding(doc: dict) -> str:
    t = _to_text(doc.get("title"))
    ing = _to_text(doc.get("ingredients"))
    ins = _to_text(doc.get("instructions") or doc.get("directions"))
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
    """Create recipes_v2 (or RECIPES_INDEX) with knn_vector embedding if missing."""
    if client.indices.exists(index=RECIPES_INDEX):
        mapping = client.indices.get_mapping(RECIPES_INDEX)
        props = mapping.get(RECIPES_INDEX, {}).get(
            "mappings", {}).get("properties", {})
        if "embedding" in props and props["embedding"].get("type") == "knn_vector":
            return
        client.indices.delete(RECIPES_INDEX)
        log.info("Recreating %s with KNN mapping", RECIPES_INDEX)

    body = {
        "settings": {"index": {"knn": True, "refresh_interval": "1s"}},
        "mappings": {"properties": {
            "id":           {"type": "keyword"},
            "title":        {"type": "text"},
            "description":  {"type": "text"},
            "ingredients":  {"type": "text"},
            "instructions": {"type": "text"},
            "photo_url":    {"type": "keyword"},
            "keto":         {"type": "boolean"},
            "vegan":        {"type": "boolean"},
            "keto_score":   {"type": "integer"},
            "vegan_score":  {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil"
                    # No 'engine', no 'parameters' on OpenSearch 3.x
                }
            }
        }}
    }
    client.indices.create(RECIPES_INDEX, body=body)
    log.info("Created index %s (KNN enabled)", RECIPES_INDEX)


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
    df = enrich_recipes_df(pd.DataFrame(records))
    enriched: List[Dict] = df.to_dict("records")
    actions: List[Dict] = []
    vocab: Set[str] = set()

    for r in enriched:
        r["embedding"] = build_doc_embedding(r)
        _id = str(r.get("id") or r.get("title") or os.urandom(8).hex())
        actions.append({"index": {"_index": RECIPES_INDEX, "_id": _id}})
        actions.append(r)
        for i in _as_list(r.get("ingredients")):
            vocab.add(normalize_ingredient(i))

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

# ── main ----------------------------------------------------------


def main(cfg):
    client = OpenSearch(hosts=[config("OPENSEARCH_URL", default=OPENSEARCH_URL)],
                        use_ssl=False, verify_certs=False, ssl_show_warn=False)
    if not wait_for_os(client):
        log.error("OpenSearch not reachable")
        sys.exit(1)

    if cfg.force:
        delete_indices(client)

    ensure_recipes_knn_index(client)
    ensure_ingredients_index(client)

    df = load_table_any(Path(cfg.data_file))
    if df.empty:
        log.warning(
            "No records loaded from %s. Indexes will exist but contain no recipes.", cfg.data_file)
        records = []
    else:
        records = df.to_dict("records")

    with tqdm(total=len(records), desc=f"Indexing into {RECIPES_INDEX}") as bar:
        for i in range(0, len(records), cfg.batch_size):
            bulk_index_recipes(
                client, records[i:i+cfg.batch_size], batch_size=cfg.batch_size)
            bar.update(min(cfg.batch_size, len(records) - i))

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
