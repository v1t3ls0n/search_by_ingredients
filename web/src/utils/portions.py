"""
Light-weight portion parser & portion-aware diet helper.

Keep it self-contained (no diet_classifiers import) so it can be reused
from anywhere without circular-import pain.
"""

from __future__ import annotations
from fractions import Fraction

from utils.usda import carbs_per_100g
from utils.text_processing import normalise, tokenize_ingredient

from utils.constants import UNIT_TO_G
from utils.patterns import _PORTION_RE

OVERFLOW_BUDGET_G = 1.0       # net-carb grams per recipe we’re willing to ignore




def _net_carbs(ingredient: str) -> float | None:
    """
    Return net-carbs (grams) contributed by *this specific quantity*,
    or None if we can’t get reliable data.
    """
    grams = _text_to_grams(ingredient)
    if grams is None:
        return None

    # Try whole-phrase USDA lookup first
    norm_name = normalise(ingredient)
    carbs = carbs_per_100g(norm_name)
    if carbs is None:
        # Fuzzy token fallback
        for tok in tokenize_ingredient(norm_name):
            carbs = carbs_per_100g(tok, fuzzy=True)
            if carbs is not None:
                break
    if carbs is None:
        return None

    return carbs * grams / 100.0  # convert to grams for this portion


def classify_portion_carb_overflow(
    ingredient: str,
    diet: str,
    base_classifier,
    overflow_budget_g: float = OVERFLOW_BUDGET_G,
) -> bool:
    """
    Keto: allow ingredient if its net-carb contribution is ≤ overflow_budget_g.
    Other diets: delegate to base_classifier.
    """
    if diet.lower() != "keto":
        return base_classifier(ingredient, diet)

    # If already keto-OK, keep it
    if base_classifier(ingredient, diet):
        return True

    carbs = _net_carbs(ingredient)
    if carbs is not None and carbs <= overflow_budget_g:
        return True

    return False


def _text_to_grams(txt: str) -> float | None:
    """“½ cup sugar” → ~120; None if we cannot parse a leading qty+unit."""
    m = _PORTION_RE.match(txt)
    if not m:
        return None
    qty_raw, unit = m.groups()
    try:
        qty = float(Fraction(qty_raw.replace(" ", "")))  # handles 1/2 etc.
    except (ValueError, ZeroDivisionError):
        return None
    unit = unit.lower()
    if unit not in UNIT_TO_G:
        return None
    return qty * UNIT_TO_G[unit]
