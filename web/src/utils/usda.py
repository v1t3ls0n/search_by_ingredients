
import os
import zipfile
import urllib.request
from typing import List, Optional, Dict
import pandas as pd

# ============================================================================
# USDA NUTRITIONAL DATABASE
# ============================================================================

# Global cache for USDA carb data
_CARB_MAP: Optional[Dict[str, float]] = None
_FUZZY_KEYS: Optional[List[str]] = None



def _download_and_extract_usda():
    """Download USDA FoodData Central if not already present."""
    cache_dir = os.path.expanduser("~/.cache/diet_classifier")
    os.makedirs(cache_dir, exist_ok=True)
    
    usda_dir = os.path.join(cache_dir, "usda")
    
    # Check if already downloaded
    if os.path.exists(os.path.join(usda_dir, "food.csv")):
        return usda_dir
    
    try:
        print("Downloading USDA nutritional database (one-time download)...")
        url = "http://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2025-04-24.zip"
        zip_path = os.path.join(cache_dir, "usda.zip")
        
        # Download
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        
        # Find extracted directory
        extracted_dir = os.path.join(cache_dir, "FoodData_Central_foundation_food_csv_2025-04-24")
        
        # Move to final location
        os.makedirs(usda_dir, exist_ok=True)
        for file in ["food.csv", "food_nutrient.csv", "nutrient.csv"]:
            src = os.path.join(extracted_dir, file)
            dst = os.path.join(usda_dir, file)
            if os.path.exists(src):
                os.rename(src, dst)
        
        # Cleanup
        os.remove(zip_path)
        if os.path.exists(extracted_dir):
            import shutil
            shutil.rmtree(extracted_dir)
        
        print("USDA database downloaded successfully!")
        return usda_dir
        
    except Exception as e:
        print(f"Warning: Could not download USDA database: {e}")
        print("Continuing with rule-based classification only...")
        return None

def _load_usda_carb_table() -> Dict[str, float]:
    """Load USDA nutritional database and extract carbohydrate content."""
    global _CARB_MAP, _FUZZY_KEYS
    
    if _CARB_MAP is not None:
        return _CARB_MAP
    
    try:
        # Try to download/locate USDA data
        usda_dir = _download_and_extract_usda()
        
        if not usda_dir:
            _CARB_MAP = {}
            _FUZZY_KEYS = []
            return _CARB_MAP
        
        # Load data files
        food_df = pd.read_csv(
            os.path.join(usda_dir, "food.csv"), 
            usecols=["fdc_id", "description"]
        )
        nutrient_df = pd.read_csv(
            os.path.join(usda_dir, "nutrient.csv"), 
            usecols=["id", "name"]
        )
        food_nutrient_df = pd.read_csv(
            os.path.join(usda_dir, "food_nutrient.csv"), 
            usecols=["fdc_id", "nutrient_id", "amount"]
        )
        
        # Find carbohydrate nutrient ID
        carb_id = int(nutrient_df.loc[
            nutrient_df["name"].str.contains("Carbohydrate, by difference", case=False), "id"
        ].iloc[0])
        
        # Get carb data
        carb_rows = food_nutrient_df[food_nutrient_df["nutrient_id"] == carb_id]
        
        # Merge with food descriptions
        carb_df = carb_rows.merge(food_df, on="fdc_id", how="left")
        carb_df = carb_df.dropna(subset=["description"])
        
        # Create mapping
        _CARB_MAP = {}
        for _, row in carb_df.iterrows():
            key = row["description"].lower().strip()
            _CARB_MAP[key] = float(row["amount"])
        
        _FUZZY_KEYS = list(_CARB_MAP.keys())
        print(f"Loaded USDA data: {len(_CARB_MAP)} food items")
        
    except Exception as e:
        print(f"Warning: Could not load USDA data: {e}")
        _CARB_MAP = {}
        _FUZZY_KEYS = []
    
    return _CARB_MAP

def carbs_per_100g(ingredient: str, fuzzy: bool = True) -> Optional[float]:
    """Look up carbohydrate content per 100g for an ingredient."""
    carb_map = _load_usda_carb_table()
    
    if not carb_map:
        return None
    
    key = ingredient.lower().strip()
    
    # Exact match
    if key in carb_map:
        return carb_map[key]
    
    # Fuzzy matching
    if fuzzy and _FUZZY_KEYS:
        try:
            # Try to use rapidfuzz if available
            from rapidfuzz import process
            match = process.extractOne(key, _FUZZY_KEYS, score_cutoff=90)
            if match:
                return carb_map.get(match[0])
        except ImportError:
            # Fallback to simple substring matching
            for usda_key in _FUZZY_KEYS:
                if key in usda_key or usda_key in key:
                    return carb_map[usda_key]
    
    return None
