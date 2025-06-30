#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keto diet classification functions.

Based on original lines 2159-2350, 1965-1989 from diet_classifiers.py
"""

import json
from typing import Iterable, Union
import numpy as np
import pandas as pd

from ..core import log, get_pipeline_state
from ..utils.constants import get_keto_patterns
from ..data.preprocessing import normalise, tokenize_ingredient
from ..data.usda import carbs_per_100g
from ..data.silver_labels import is_keto_ingredient_list


def is_ingredient_keto(ingredient: str) -> bool:
    """
    Determine if a single ingredient is keto-friendly.

    Implements a comprehensive decision pipeline:

    1. **Whitelist Check**: Immediate acceptance for known keto ingredients
    2. **Numeric Rule**: Reject if carbs > 10g/100g (USDA database)
       - Whole phrase lookup
       - Token-level fallback (ignoring stop words)
    3. **Regex Blacklist**: Fast pattern matching against NON_KETO
    4. **Token Blacklist**: Detailed token-level analysis
    5. **ML Model**: Machine learning prediction with rule verification

    Args:
        ingredient: Raw ingredient string

    Returns:
        True if keto-friendly, False otherwise

    Example:
        >>> is_ingredient_keto("almond flour")
        True  # Whitelisted
        >>> is_ingredient_keto("white rice")
        False  # High carb content
        >>> is_ingredient_keto("banana")
        False  # In NON_KETO list
        
    Based on original lines 2159-2305
    """
    if not ingredient:
        return True

    patterns = get_keto_patterns()
    
    # 1. Whitelist (immediate accept)
    if patterns['whitelist'].search(ingredient):
        return True

    # 2. Numeric carbohydrate rule
    norm = normalise(ingredient)

    # 2a. Whole-phrase lookup
    carbs = carbs_per_100g(norm)
    if carbs is not None:
        return carbs <= 10.0

    # 2b. Token-level fallback
    for tok in tokenize_ingredient(norm):
        # Skip common stop words and units
        if tok in {"raw", "fresh", "dried", "powder", "mix", "sliced",
                   "organic", "cup", "cups", "tsp", "tbsp", "g", "kg", "oz"}:
            continue
        carbs_tok = carbs_per_100g(tok, fuzzy=True)
        if carbs_tok is not None and carbs_tok > 10.0:
            return False

    # 3. Regex blacklist (fast)
    if patterns['blacklist'].search(norm):
        return False

    # 4. Token-level heuristic list
    if not is_keto_ingredient_list(tokenize_ingredient(norm)):
        return False

    # If all rules pass, it's keto
    return True


def is_keto(ingredients: Union[Iterable[str], str]) -> bool:
    """
    Check if all ingredients in a recipe are keto-friendly.

    Uses the best available method:
    1. Trained ML model (if available) - evaluates the full recipe
    2. Falls back to rule-based classification per ingredient

    Args:
        ingredients: Either a string (comma-separated or JSON) or an iterable

    Returns:
        True if keto-friendly, False otherwise

    Example:
        >>> is_keto("almond flour, eggs, butter")
        True
        >>> is_keto(["chicken", "broccoli", "rice"])
        False  # Rice is not keto
        >>> is_keto('["spinach", "cheese", "cream"]')
        True
        
    Based on original lines 2306-2350
    """
    # Parse ingredients to string format
    if isinstance(ingredients, str):
        ingredients_str = ingredients
        try:
            if ingredients.startswith('['):
                ingredients_list = json.loads(ingredients)
            else:
                ingredients_list = [i.strip() for i in ingredients.split(',') if i.strip()]
        except Exception:
            ingredients_list = [ingredients]
    else:
        ingredients_list = list(ingredients)
        ingredients_str = ', '.join(str(i) for i in ingredients_list)
    
    # Check if we have a trained model
    pipeline_state = get_pipeline_state()
    pipeline_state.ensure_pipeline_initialized()
    
    model = pipeline_state.get_best_model('keto')
    
    if model is not None and pipeline_state.vectorizer:
        # Use ML model on the full recipe
        try:
            # Normalize the full ingredient text
            norm = normalise(ingredients_str)
            X = pipeline_state.vectorizer.transform([norm])
            
            # Check if model expects more features (was trained with images)
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                actual_features = X.shape[1]
                
                if expected_features > actual_features:
                    # Model expects images, pad with zeros
                    from ..features.combiners import combine_features
                    padding = np.zeros((1, 2048), dtype=np.float32)
                    X = combine_features(X, padding)
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[0, 1]
            else:
                prob = float(model.predict(X)[0])
            
            # Apply rule-based verification
            from ..classification.verification import verify_with_rules
            prob_adj = verify_with_rules("keto", pd.Series([norm]), np.array([prob]))[0]
            
            return prob_adj >= 0.5
            
        except Exception as e:
            log.debug(f"ML prediction failed for keto: {e}, falling back to rules")
    
    # Fall back to rule-based classification (check each ingredient)
    return all(is_ingredient_keto(ing) for ing in ingredients_list)


def find_non_keto_hits(text: str) -> list[str]:
    """
    Find all non-keto ingredients present in the text.

    Used for debugging and understanding why an ingredient
    was classified as non-keto.

    Args:
        text: Normalized ingredient text

    Returns:
        Sorted list of matching non-keto ingredients
        
    Based on original lines 1990-2002
    """
    from ..utils.constants import NON_KETO
    
    tokens = set(tokenize_ingredient(text))
    return sorted([
        ingredient for ingredient in NON_KETO
        if all(tok in tokens for tok in ingredient.split())
    ])