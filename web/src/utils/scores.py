
from utils.classifiers import classify_ingredient
from utils.portions import classify_portion_carb_overflow 

# ---------------------------------------------------------------------------
# KETO / VEGAN  PERCENTAGE-SCORE HELPERS
# ---------------------------------------------------------------------------

def _diet_score_portion_aware(ingredients: list[str], diet: str) -> int:
    """
    Uses classify_portion_carb_overflow -> applies the 1-g net-carb rule
    for keto; falls back to strict check for vegan.
    """
    if len(ingredients) == 0:
        return 0
    ok = sum(classify_portion_carb_overflow(ing, diet, classify_ingredient) for ing in ingredients)
    return round(100 * ok / len(ingredients))


def _diet_score_simple(ingredients: list[str], diet: str) -> int:
    """
    Lightweight score: % of ingredients that pass the *strict* classifier.
    Ignores quantities and USDA look-ups entirely.
    """
    if len(ingredients) == 0:
        return 0
    passed = sum(classify_ingredient(ing, diet) for ing in ingredients)
    return round(100 * passed / len(ingredients))
