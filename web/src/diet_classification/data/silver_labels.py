#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silver label generation using rule-based heuristics.

Based on original lines 3302-3373 from diet_classifiers.py
"""

import pandas as pd

from ..utils.constants import get_keto_patterns, get_vegan_patterns
from .preprocessing import normalise, tokenize_ingredient
from .usda import carbs_per_100g


def is_keto_ingredient_list(tokens: list[str]) -> bool:
    """
    Check if a tokenized ingredient list is keto-friendly.

    Performs token-level matching against the NON_KETO list.
    Returns False if all tokens of any non-keto ingredient are found.

    Args:
        tokens: List of normalized ingredient tokens

    Returns:
        True if keto-friendly, False otherwise
        
    Based on original lines 1965-1989
    """
    from ..utils.constants import NON_KETO
    
    for ingredient in NON_KETO:
        ing_tokens = ingredient.split()
        if all(tok in tokens for tok in ing_tokens):
            return False
    return True


def _rule_only_keto(text: str) -> bool:
    """
    Apply keto rules without ML model fallback.

    Used for silver label generation where we want consistent
    rule-based labels without ML influence.
    
    Based on original lines 3302-3327
    """
    patterns = get_keto_patterns()
    
    if patterns['whitelist'].search(text):
        return True
    norm = normalise(text)
    c = carbs_per_100g(norm)
    if c is not None and c > 10:
        return False
    if patterns['blacklist'].search(norm):
        return False
    if not is_keto_ingredient_list(tokenize_ingredient(norm)):
        return False
    return True


def _rule_only_vegan(text: str) -> bool:
    """
    Apply vegan rules without ML model fallback.

    Used for silver label generation where we want consistent
    rule-based labels without ML influence.
    
    Based on original lines 3329-3349
    """
    patterns = get_vegan_patterns()
    
    if patterns['whitelist'].search(text):
        return True
    norm = normalise(text)
    if patterns['blacklist'].search(norm) and not patterns['whitelist'].search(text):
        return False
    return True


def build_silver(recipes: pd.DataFrame) -> pd.DataFrame:
    """
    Generate silver labels for recipe dataset using heuristic rules.

    Creates weak labels that can be used for training when manual
    labels are unavailable. The silver labels are less accurate than
    gold standard labels but enable training on much larger datasets.

    Args:
        recipes: DataFrame with 'ingredients' column

    Returns:
        DataFrame with added columns:
        - clean: Normalized ingredient text
        - silver_keto: Binary keto label (0 or 1)
        - silver_vegan: Binary vegan label (0 or 1)
        
    Based on original lines 3351-3373
    """
    df = recipes[["ingredients"]].copy()
    df["clean"] = df["ingredients"].fillna("").map(normalise)

    df["silver_keto"] = df["clean"].map(lambda t: int(_rule_only_keto(t)))
    df["silver_vegan"] = df["clean"].map(lambda t: int(_rule_only_vegan(t)))

    return df