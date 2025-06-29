#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vegan diet classification functions.

Based on original lines 2352-2443 from diet_classifiers.py
"""

import json
from typing import Iterable, Union
import numpy as np

from ..core import log, get_pipeline_state
from ..utils.constants import get_vegan_patterns
from ..data.preprocessing import normalise


def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Determine if an ingredient is vegan.

    Uses a simplified pipeline compared to keto classification:

    1. **Whitelist Check**: Accept known vegan alternatives
    2. **Blacklist Check**: Reject animal products
    3. **ML Model**: Machine learning with verification

    Args:
        ingredient: Raw ingredient string

    Returns:
        True if vegan, False otherwise

    Example:
        >>> is_ingredient_vegan("almond milk")
        True  # Whitelisted plant milk
        >>> is_ingredient_vegan("chicken")
        False  # Animal product
        >>> is_ingredient_vegan("honey")
        False  # Animal product (bee-derived)
        
    Based on original lines 2352-2396
    """
    if not ingredient:
        return True

    patterns = get_vegan_patterns()
    
    # Quick whitelist check
    if patterns['whitelist'].search(ingredient):
        return True

    # Normalize
    normalized = normalise(ingredient)

    # Quick blacklist check
    if patterns['blacklist'].search(normalized) and not patterns['whitelist'].search(ingredient):
        return False

    # Use ML model if available
    pipeline_state = get_pipeline_state()
    pipeline_state.ensure_pipeline_initialized()
    
    if 'vegan' in pipeline_state.models and pipeline_state.initialized:
        model = pipeline_state.models['vegan']
        if pipeline_state.vectorizer:
            try:
                X = pipeline_state.vectorizer.transform([normalized])

                # Check if model expects more features (was trained with images)
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    actual_features = X.shape[1]

                    if expected_features > actual_features:
                        # Model expects images, pad with zeros
                        from ..features.combiners import combine_features
                        padding = np.zeros((1, 2048), dtype=np.float32)
                        X = combine_features(X, padding)

                prob = model.predict_proba(X)[0, 1]
                
                # Apply rule-based verification
                from ..classification.verification import verify_with_rules
                import pandas as pd
                prob_adj = verify_with_rules(
                    "vegan", pd.Series([normalized]), np.array([prob]))[0]
                return prob_adj >= 0.5
                
            except Exception as e:
                log.warning(
                    "Vectorizer failed: %s. Using rule-based fallback.", e)
                # Use rule-based approach as fallback
                return True  # Passed whitelist/blacklist checks

    return True


def is_vegan(ingredients: Union[Iterable[str], str]) -> bool:
    """
    Check if all ingredients in a recipe are vegan.

    This is the main public API for vegan classification. It handles
    various input formats and ensures all ingredients pass the vegan test.

    Args:
        ingredients: Either a string (comma-separated or JSON) or an iterable

    Returns:
        True if ALL ingredients are vegan, False otherwise

    Example:
        >>> is_vegan("tofu, soy sauce, vegetables")
        True
        >>> is_vegan(["almond milk", "honey", "oats"])
        False  # Honey is not vegan
        >>> is_vegan('["rice", "beans", "vegetables"]')
        True
        
    Based on original lines 2398-2443
    """
    if isinstance(ingredients, str):
        try:
            if ingredients.startswith('['):
                ingredients = json.loads(ingredients)
            else:
                ingredients = [i.strip()
                               for i in ingredients.split(',') if i.strip()]
        except Exception:
            ingredients = [ingredients]
    return all(is_ingredient_vegan(ing) for ing in ingredients)