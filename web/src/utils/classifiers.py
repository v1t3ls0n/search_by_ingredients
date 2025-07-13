from utils.constants import NON_KETO, NON_VEGAN
from utils.usda import carbs_per_100g
from utils.text_processing import normalise, tokenize_ingredient
from utils.patterns import RX_KETO, RX_WL_KETO, RX_VEGAN, RX_WL_VEGAN




# ============================================================================
# MAIN CLASSIFICATION FUNCTIONS
# ============================================================================


def classify_ingredient(ingredient: str, task: str) -> bool:
    """
    Unified classification function for both keto and vegan.

    Args:
        ingredient: The ingredient to classify
        task: Either "keto" or "vegan"

    Decision pipeline:
    1. Whitelist Check: Immediate acceptance for known friendly ingredients
    2. Normalize the ingredient  
    3. Whitelist Check Again: Check normalized form against whitelist
    4. Regex Blacklist: Pattern matching against blacklist (hard rules)
    5. Token-level Blacklist: Multi-word ingredient detection
    6. Task-specific analysis: USDA for keto, single-token check for vegan
    """
    if not ingredient:
        return True

    # Select appropriate patterns and lists based on task
    if task == "keto":
        whitelist_rx = RX_WL_KETO
        blacklist_rx = RX_KETO
        blacklist_items = NON_KETO
    else:  # vegan
        whitelist_rx = RX_WL_VEGAN
        blacklist_rx = RX_VEGAN
        blacklist_items = NON_VEGAN

    # 1. Whitelist (immediate accept) - Check ORIGINAL string
    if whitelist_rx.search(ingredient):
        return True

    # 2. Normalize
    norm = normalise(ingredient)

    # 3. Whitelist check on NORMALIZED string
    if whitelist_rx.search(norm):
        return True

    # 4. Regex blacklist (hard domain rules)
    if blacklist_rx.search(norm):
        return False

    # 5. Token-level blacklist check
    tokens_set = set(tokenize_ingredient(norm))
    for blacklist_item in blacklist_items:
        ing_tokens = blacklist_item.split()
        if all(tok in tokens_set for tok in ing_tokens):
            return False

    # 6. Task-specific token analysis
    if task == "keto":
        # USDA Numeric carbohydrate rule
        # Whole-phrase lookup
        carbs = carbs_per_100g(norm)
        if carbs is not None:
            return carbs <= 10.0

    # Token-level analysis for both tasks
    tokens = tokenize_ingredient(norm)
    for tok in tokens:
        # Skip common stop words and units
        if tok in {"raw", "fresh", "dried", "powder", "mix", "sliced",
                   "organic", "cup", "cups", "tsp", "tbsp", "g", "kg", "oz"}:
            continue

        if task == "keto":
            # USDA token-level fallback
            carbs_tok = carbs_per_100g(tok, fuzzy=True)
            if carbs_tok is not None and carbs_tok > 10.0:
                return False
        else:  # vegan
            # Check if individual token is an animal product
            if tok in blacklist_items:
                return False

    # 7. Default to friendly if no conflicts found
    return True


def is_ingredient_keto(ingredient: str) -> bool:
    """Determine if a single ingredient is keto-friendly."""
    return classify_ingredient(ingredient, "keto")


def is_ingredient_vegan(ingredient: str) -> bool:
    """Determine if an ingredient is vegan."""
    return classify_ingredient(ingredient, "vegan")


