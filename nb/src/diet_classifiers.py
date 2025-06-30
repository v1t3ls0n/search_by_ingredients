import json
import sys
import re
import unicodedata
import os
import zipfile
import urllib.request
from argparse import ArgumentParser
from typing import List, Optional, Dict
from time import time
import pandas as pd

try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")

# ============================================================================
# CONSTANTS AND PATTERNS
# ============================================================================

# High-carb ingredients that disqualify keto classification
NON_KETO = [
    # High-carb fruits
    "apple", "banana", "orange", "grape", "kiwi", "mango", "peach",
    "strawberry", "pineapple", "apricot", "tangerine", "persimmon",
    "pomegranate", "prune", "papaya", "jackfruit",
    # Grains and grain products
    "white rice", "long grain rice", "cornmeal", "corn",
    "all-purpose flour", "bread", "pasta", "couscous",
    "bulgur", "quinoa", "barley flour", "buckwheat flour",
    "durum wheat flour", "wheat flour", "whole-wheat flour",
    "oat flour", "oatmeal", "rye flour", "semolina",
    # Starchy vegetables
    "potato", "baking potato", "potato wedge", "potato slice",
    "russet potato", "sweet potato", "yam", "cassava",
    # Legumes
    "kidney bean", "black bean", "pinto bean", "navy bean",
    "lima bean", "cannellini bean", "great northern bean",
    "garbanzo bean", "chickpea", "adzuki bean", "baked bean",
    # Sweeteners and sugars
    "sugar", "brown sugar", "coconut sugar", "muscovado sugar",
    "demerara", "turbinado", "molasses", "honey", "agave nectar",
    "maple syrup",
    # Beverages
    "ale", "beer", "lager", "ipa", "pilsner", "stout", "porter",
]

# Animal-derived ingredients that disqualify vegan classification
NON_VEGAN = [
    # Meat - Red meat
    'beef', 'steak', 'ribeye', 'sirloin', 'veal', 'lamb', 'mutton',
    'pork', 'bacon', 'ham', 'boar', 'goat', 'kid', 'venison',
    'rabbit', 'hare',
    # Meat - Poultry
    'chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
    # Meat - Organ meats
    'liver', 'kidney', 'heart', 'tongue', 'brain', 'sweetbread',
    'tripe', 'gizzard', 'offal', 'bone', 'marrow', 'oxtail',
    # Fish and Seafood
    'fish', 'salmon', 'tuna', 'cod', 'haddock', 'halibut',
    'mackerel', 'herring', 'sardine', 'anchovy', 'trout',
    'shrimp', 'prawn', 'crab', 'lobster', 'langoustine',
    'clam', 'mussel', 'oyster', 'scallop', 'squid', 'calamari',
    # Dairy
    'milk', 'cream', 'butter', 'buttermilk', 'condensed', 'evaporated',
    'lactose', 'whey', 'casein', 'ghee', 'kefir',
    'cheese', 'cheddar', 'mozzarella', 'parmesan', 'parmigiano',
    'yogurt', 'sourcream', 'cremefraiche', 'curd', 'custard',
    # Eggs
    'egg', 'yolk', 'albumen', 'omelet', 'omelette', 'meringue',
    # Other animal products
    'honey', 'shellfish', 'escargot', 'snail', 'frog',
]

# Regex patterns for keto-friendly ingredients (overrides blacklist)
KETO_WHITELIST_PATTERNS = [
    r"\balmond flour\b", r"\bcoconut flour\b", r"\bflaxseed flour\b",
    r"\bchia flour\b", r"\bkidney\b", r"\blemon juice\b",
    r"\balmond milk\b", r"\bcoconut milk\b", r"\bstevia\b",
    r"\berytritol\b", r"\bmonk fruit\b", r"\bsugar[- ]free\b",
]

# Regex patterns for vegan-friendly ingredients (overrides blacklist)
VEGAN_WHITELIST_PATTERNS = [
    r"\beggplant\b", r"\bmillet\b", r"\bcoconut\s+milk\b",
    r"\boat\s+milk\b", r"\bsoy\s+milk\b", r"\balmond\s+milk\b",
    r"\bbutternut\b", r"\bcocoa\s+butter\b", r"\bpeanut\s+butter\b",
    r"\balmond\s+butter\b", r"\bvegan\s+butter\b", r"\bhoneydew\b",
    r"\bcoconut\s+cream\b", r"\bcream\s+of\s+tartar\b",
]

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
        url = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2025-04-24.zip"
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

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def tokenize_ingredient(text: str) -> List[str]:
    """Extract word tokens from ingredient text."""
    return re.findall(r"\b\w[\w-]*\b", text.lower())

def normalise(t: str) -> str:
    """Normalize ingredient text for consistent matching."""
    if not isinstance(t, str):
        t = str(t)
    
    # Unicode normalization - remove accents
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    
    # Remove parenthetical content and convert to lowercase
    t = re.sub(r"\([^)]*\)", " ", t.lower())
    
    # Remove units of measurement
    units = re.compile(r"\b(?:g|gram|kg|oz|ml|l|cup|cups|tsp|tbsp|teaspoon|"
                      r"tablespoon|pound|lb|slice|slices|small|large|medium)\b")
    t = units.sub(" ", t)
    
    # Remove numbers (including fractions)
    t = re.sub(r"\d+(?:[/\.]\d+)?", " ", t)
    
    # Remove punctuation
    t = re.sub(r"[^\w\s-]", " ", t)
    
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    
    return t

# ============================================================================
# PATTERN COMPILATION
# ============================================================================

def compile_any(words: List[str]) -> re.Pattern:
    """Compile a list of words into a single regex pattern."""
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, words)), re.I)

# Compile patterns once
RX_KETO = compile_any(NON_KETO)
RX_WL_KETO = re.compile("|".join(KETO_WHITELIST_PATTERNS), re.I)
RX_VEGAN = compile_any(NON_VEGAN)
RX_WL_VEGAN = re.compile("|".join(VEGAN_WHITELIST_PATTERNS), re.I)

# ============================================================================
# MAIN CLASSIFICATION FUNCTIONS
# ============================================================================

def is_ingredient_keto(ingredient: str) -> bool:
    """
    Determine if a single ingredient is keto-friendly.
    
    Decision pipeline:
    1. Whitelist Check: Immediate acceptance for known keto ingredients
    2. USDA Numeric Rule: Reject if carbs > 10g/100g
    3. Regex Blacklist: Pattern matching against NON_KETO
    4. Token Blacklist: Token-level analysis
    """
    if not ingredient:
        return True
    
    # 1. Whitelist (immediate accept)
    if RX_WL_KETO.search(ingredient):
        return True
    
    # 2. Numeric carbohydrate rule
    norm = normalise(ingredient)
    
    # Whole-phrase lookup
    carbs = carbs_per_100g(norm)
    if carbs is not None:
        return carbs <= 10.0
    
    # Token-level fallback
    tokens = tokenize_ingredient(norm)
    for tok in tokens:
        # Skip common stop words
        if tok in {"raw", "fresh", "dried", "powder", "mix", "sliced", "organic"}:
            continue
        carbs_tok = carbs_per_100g(tok, fuzzy=True)
        if carbs_tok is not None and carbs_tok > 10.0:
            return False
    
    # 3. Regex blacklist
    if RX_KETO.search(norm):
        return False
    
    # 4. Token-level heuristic
    tokens_set = set(tokens)
    for non_keto in NON_KETO:
        ing_tokens = non_keto.split()
        if all(tok in tokens_set for tok in ing_tokens):
            return False
    
    return True

def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Determine if an ingredient is vegan.
    
    Decision pipeline:
    1. Whitelist Check: Accept known vegan alternatives
    2. Blacklist Check: Reject animal products
    """
    if not ingredient:
        return True
    
    # Quick whitelist check
    if RX_WL_VEGAN.search(ingredient):
        return True
    
    # Normalize
    normalized = normalise(ingredient)
    
    # Quick blacklist check
    if RX_VEGAN.search(normalized) and not RX_WL_VEGAN.search(ingredient):
        return False
    
    return True

def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, ingredients))

def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, ingredients))


def main(args):
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    try:
        start_time = time()
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(
            is_vegan)

        end_time = time()
    except Exception as e:
        print(f"Error: {e}")
        return -1

    print("===Keto===")
    print(classification_report(
        ground_truth['keto'], ground_truth['keto_pred']))
    print("===Vegan===")
    print(classification_report(
        ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time} seconds ==")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str,
                        default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))
