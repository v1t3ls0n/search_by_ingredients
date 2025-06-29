#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USDA nutritional database handling for carbohydrate-based keto classification.

Based on original lines 926-1020, 2055-2157 from diet_classifiers.py
"""

from typing import Optional
import pandas as pd
from rapidfuzz import process

from ..core import log, get_pipeline_state
from ..config import CFG
from .preprocessing import normalise


def _load_usda_carb_table() -> pd.DataFrame:
    """
    Load USDA nutritional database and extract carbohydrate content.

    This function loads the USDA FoodData Central database files and creates
    a lookup table mapping food descriptions to carbohydrate content per 100g.
    Used for the numeric keto rule (foods with >10g carbs/100g are non-keto).

    Returns:
        DataFrame with columns:
        - food_desc: Lowercase food description
        - carb_100g: Carbohydrate content per 100g

    Note:
        Requires USDA database CSV files in the configured directory:
        - food.csv: Food items and descriptions
        - nutrient.csv: Nutrient definitions
        - food_nutrient.csv: Nutrient content per food item
        
    Based on original lines 926-1020
    """
    # 1. Resolve file paths
    usda = CFG.usda_dir
    food_csv = usda / "food.csv"
    food_nutrient_csv = usda / "food_nutrient.csv"
    nutrient_csv = usda / "nutrient.csv"

    if not (food_csv.exists() and food_nutrient_csv.exists() and nutrient_csv.exists()):
        log.warning(
            "USDA tables not found in %s – skipping numeric carb table", usda)
        return pd.DataFrame(columns=["food_desc", "carb_100g"])

    # 2. Locate nutrient_id for carbohydrate
    nutrient = pd.read_csv(nutrient_csv, usecols=["id", "name"])
    carb_id = int(
        nutrient.loc[
            nutrient["name"].str.contains(
                "Carbohydrate, by difference", case=False),
            "id"
        ].iloc[0]
    )

    # 3. Pull carb rows from food_nutrient
    carb_rows = pd.read_csv(
        food_nutrient_csv,
        usecols=["fdc_id", "nutrient_id", "amount"],
        dtype={"fdc_id": "int32", "nutrient_id": "int16", "amount": "float32"},
    ).query("nutrient_id == @carb_id")[["fdc_id", "amount"]]

    # 4. Join with food descriptions
    food = pd.read_csv(
        food_csv,
        usecols=["fdc_id", "description"],
        dtype={"fdc_id": "int32", "description": "string"},
    )
    carb_df = (
        carb_rows.merge(food, on="fdc_id", how="left", validate="m:1")
        .dropna(subset=["description"])
        .assign(
            food_desc=lambda df: df["description"].str.lower().str.strip(),
            carb_100g=lambda df: df["amount"].round(1),
        )[["food_desc", "carb_100g"]]
        .drop_duplicates("food_desc")
        .reset_index(drop=True)
    )
    log.info("USDA carb table loaded: %d distinct food descriptions", len(carb_df))
    return carb_df


def label_usda_keto_data(carb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the USDA carb table into silver-style training data.

    Labels each ingredient as keto if carbs_per_100g < 10.
    Applies full ingredient normalization using `normalise()`.

    Args:
        carb_df: Output of _load_usda_carb_table()

    Returns:
        DataFrame with: ingredient, clean, silver_keto, silver_vegan, source
        
    Based on original lines 1021-1066
    """
    df = carb_df.copy()
    df["ingredient"] = df["food_desc"]
    df["clean"] = df["ingredient"].map(normalise)
    df["silver_keto"] = (df["carb_100g"] < 10).astype(int)
    df["silver_vegan"] = pd.NA  # Use pandas NA for missing values
    df["source"] = "usda"
    return df[["ingredient", "clean", "silver_keto", "silver_vegan", "source"]]


def ensure_carb_map() -> None:
    """
    Lazy-load USDA table into state for fast lookups.
    
    This replaces the global _ensure_carb_map() function.
    Based on original lines 2055-2068
    """
    pipeline_state = get_pipeline_state()
    
    if pipeline_state.carb_map is None:
        df = _load_usda_carb_table()
        pipeline_state.carb_map = df.set_index("food_desc")["carb_100g"].to_dict()
        pipeline_state.fuzzy_keys = list(pipeline_state.carb_map)
        log.info("Carb map initialised (%d keys)", len(pipeline_state.carb_map))


def carbs_per_100g(ing: str, fuzzy: bool = True) -> Optional[float]:
    """
    Look up carbohydrate content per 100g for an ingredient.

    First attempts exact matching, then falls back to fuzzy matching
    using RapidFuzz library with a similarity threshold of 90%.

    Args:
        ing: Normalized ingredient string
        fuzzy: Whether to use fuzzy matching if exact match fails

    Returns:
        Carbohydrate grams per 100g, or None if not found

    Example:
        >>> carbs_per_100g("white rice")
        28.2
        >>> carbs_per_100g("wite rice")  # Fuzzy match
        28.2
        
    Based on original lines 2125-2157
    """
    ensure_carb_map()
    pipeline_state = get_pipeline_state()
    
    key = ing.lower().strip()
    val = pipeline_state.carb_map.get(key)
    if val is not None or not fuzzy:
        return val

    # Fuzzy matching fallback
    match = process.extractOne(key, pipeline_state.fuzzy_keys, score_cutoff=90)
    return pipeline_state.carb_map.get(match[0]) if match else None