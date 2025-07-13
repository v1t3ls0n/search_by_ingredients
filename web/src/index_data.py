#!/usr/bin/env python
"""
Bulk-index Allrecipes data into OpenSearch *with diet metadata*.

Usage
-----
docker compose exec web \
    python web/index_data.py --force   # re-create indices from scratch
"""
from __future__ import annotations

import logging
import re
import string
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import sleep
from typing import Dict, List,Set

import pandas as pd
from opensearchpy import OpenSearch
from tqdm import tqdm

# Try to import swifter for parallel processing
try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False

# ── local helpers (diet classification) ───────────────────────────
from diet_classifiers import is_keto, is_vegan, diet_score  # noqa: E402

# ── logging ───────────────────────────────────────────────────────
logging.getLogger("opensearch").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────


def normalize_ingredient(txt: str) -> str:
    """Very loose normaliser so '2 cups strawberries' → 'strawberry'."""
    if not isinstance(txt, str):
        return str(txt)

    txt = txt.lower().strip()
    txt = txt.rsplit(",", 1)[0]                 # trailing comments
    txt = re.sub(r"\([^()]+\)", "", txt)        # (...) groups
    txt = re.sub(r"\d+\s*\d*/\d*", "", txt)     # 1 1/2, 200
    txt = txt.translate({ord(c): " " for c in string.punctuation})

    units = (
        "cup cups can cans tablespoon tablespoons tbsp teaspoon teaspoons tsp "
        "ounce ounces oz pound pounds lb lbs gram grams g kilogram kilograms kg "
        "milliliter milliliters ml liter liters l pinch pinches dash dashes "
        "piece pieces slice slices small medium large cube cubes inch inches "
        "cm mm quart quarts qt jar scoop scoops gallon gallons gal pint pints "
        "pt fluid ounce fluid ounces fl oz package packages pkg pack packs"
    ).split()
    txt = re.sub(rf"\b({'|'.join(units)})\b", "", txt)

    # crude plural → singular
    if txt.endswith("ies"):
        txt = txt[:-3] + "y"
    elif txt.endswith("es") and txt[-3] in "sxz":
        txt = txt[:-2]
    elif txt.endswith("s"):
        txt = txt[:-1]

    return " ".join(txt.split())



def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add keto/vegan flags and scores to a DataFrame in parallel (swifter)
    or sequentially (plain .apply) depending on availability.
    """
    if SWIFTER_AVAILABLE:
        log.info("Applying diet classifiers with swifter (parallel)…")
        apply = lambda s: s.swifter.apply            
    else:
        log.warning("Swifter not available; using standard pandas")
        log.info("Install swifter for parallel processing:  pip install swifter")
        apply = lambda s: s.apply                    

    df["keto"]        = apply(df["ingredients"])(is_keto)
    df["vegan"]       = apply(df["ingredients"])(is_vegan)
    df["keto_score"]  = apply(df["ingredients"])(lambda ings: diet_score(ings, "keto"))
    df["vegan_score"] = apply(df["ingredients"])(lambda ings: diet_score(ings, "vegan"))
    return df


# ── OpenSearch helpers ────────────────────────────────────────────
def wait_for_os(client: OpenSearch, retries: int = 30, delay: int = 2) -> bool:
    for _ in range(retries):
        if client.ping():
            return True
        sleep(delay)
    return False


def delete_indices(client: OpenSearch) -> None:
    for idx in ("recipes", "ingredients"):
        if client.indices.exists(index=idx):
            client.indices.delete(index=idx)
            log.info("Deleted index %s", idx)


def create_indices(client: OpenSearch) -> None:
    if not client.indices.exists(index="recipes"):
        client.indices.create(
            index="recipes",
            body={
                "mappings": {
                    "properties": {
                        "title":        {"type": "text"},
                        "description":  {"type": "text"},
                        "ingredients":  {"type": "text"},
                        "instructions": {"type": "text"},
                        "photo_url":    {"type": "keyword"},
                        # diet metadata
                        "keto":        {"type": "boolean"},
                        "vegan":       {"type": "boolean"},
                        "keto_score":  {"type": "integer"},
                        "vegan_score": {"type": "integer"},
                    }
                }
            },
        )
        log.info("Created index recipes")

    if not client.indices.exists(index="ingredients"):
        client.indices.create(
            index="ingredients",
            body={"mappings": {"properties": {"ingredients": {"type": "text"}}}},
        )
        log.info("Created index ingredients")


def batch_index(
    client: OpenSearch,
    recipes: List[Dict],
    batch_size: int = 10_240,
) -> None:
    """
    Bulk-index recipe documents (plus a flat ingredients vocabulary) into OpenSearch.
    Diet flags / scores are added in parallel via swifter when possible.

    Parameters
    ----------
    client      : an `opensearchpy.OpenSearch` instance
    recipes     : list of recipe dicts; each dict *must* contain `"ingredients"` (list[str])
    batch_size  : number of recipe docs per bulk chunk (default: 10 240)
    """
    # 1️⃣  Enrich recipes (parallel if swifter):
    df = _enrich_df(pd.DataFrame(recipes))
    enriched: List[Dict] = df.to_dict("records")

    # 2️⃣  Bulk-index recipes + build vocab on the fly:
    actions: List[Dict] = []
    vocab: Set[str] = set()

    for r in enriched:
        actions += [{"index": {"_index": "recipes"}}, r]
        vocab.update(normalize_ingredient(i) for i in r["ingredients"])

        if len(actions) >= batch_size * 2:           # each doc gets two lines
            client.bulk(actions)
            actions.clear()

    if actions:
        client.bulk(actions)

    # 3️⃣  One more bulk pass for unique ingredient tokens:
    if vocab:
        ing_actions: List[Dict] = []
        for token in vocab:
            ing_actions += [
                {"index": {"_index": "ingredients"}},
                {"ingredients": token},
            ]
        client.bulk(ing_actions)

# ── main ──────────────────────────────────────────────────────────
def main(cfg):
    client = OpenSearch(
        hosts=[cfg.opensearch_url],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )

    if not wait_for_os(client):
        log.error("OpenSearch not reachable"); sys.exit(1)

    if cfg.force:
        delete_indices(client)
    elif client.indices.exists(index="recipes"):
        log.info("Data already present – skipping (use --force to re-index)")
        return

    create_indices(client)

    df = pd.read_parquet(cfg.data_file)
    records = df.to_dict("records")

    with tqdm(total=len(records), desc="Indexing recipes") as bar:
        for i in range(0, len(records), cfg.batch_size):
            batch_index(client, records[i : i + cfg.batch_size])
            bar.update(cfg.batch_size)

    log.info("✔ Done – %s recipes indexed", len(records))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument(
        "--data_file", type=Path,
        default=Path("data/allrecipes.parquet"),
        help="Parquet file with recipes",
    )
    p.add_argument(
        "--opensearch_url", default="http://localhost:9200",
        help="OpenSearch URL",
    )
    p.add_argument("--force", action="store_true", help="Delete & re-index data")
    main(p.parse_args())
