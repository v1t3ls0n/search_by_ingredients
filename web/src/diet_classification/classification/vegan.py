#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vegan diet classification functions.

Based on original lines 2352-2443 from diet_classifiers.py
"""

import json
from typing import Iterable, Union, Optional
import numpy as np
import pandas as pd

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

    return True


def is_vegan(ingredients: Union[Iterable[str], str], mode: Optional[str] = None) -> bool:
    """
    Check if all ingredients in a recipe are vegan.

    Uses the best available method for the specified mode:
    1. Mode-specific trained ML model (if available)
    2. General best model (if no mode-specific model)
    3. Falls back to rule-based classification per ingredient

    Args:
        ingredients: Either a string (comma-separated or JSON) or an iterable
        mode: Optional mode ('text', 'image', 'both') to use specific model

    Returns:
        True if ALL ingredients are vegan, False otherwise

    Example:
        >>> is_vegan("tofu, soy sauce, vegetables")
        True
        >>> is_vegan(["almond milk", "honey", "oats"])
        False  # Honey is not vegan
        >>> is_vegan('["rice", "beans", "vegetables"]')
        True
        >>> is_vegan("tofu, vegetables", mode="text")  # Use text-only model
        True
        
    Based on original lines 2398-2443
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
    
    # Try to get mode-specific model first
    model = None
    if mode:
        from ..models.io import load_best_model_for_mode
        model = load_best_model_for_mode('vegan', mode)
        if model:
            log.debug(f"Using {mode}-specific model for vegan classification")
    
    # Fall back to general best model
    if model is None:
        model = pipeline_state.get_best_model('vegan')
        if model:
            log.debug(f"Using general best model for vegan classification")
    
    if model is not None and pipeline_state.vectorizer:
        # Use ML model on the full recipe
        try:
            # Normalize the full ingredient text
            normalized = normalise(ingredients_str)
            X = pipeline_state.vectorizer.transform([normalized])
            
            # Check if model expects more features
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                actual_features = X.shape[1]
                
                if expected_features > actual_features:
                    # Model expects images
                    if mode == 'image':
                        log.warning("Image model requested but no image features available")
                        # Fall back to rules
                        return all(is_ingredient_vegan(ing) for ing in ingredients_list)
                    elif mode == 'both' or (mode is None and expected_features > actual_features + 2048):
                        # Pad with zeros for missing image features
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
            prob_adj = verify_with_rules("vegan", pd.Series([normalized]), np.array([prob]))[0]
            
            return prob_adj >= 0.5
            
        except Exception as e:
            log.debug(f"ML prediction failed for vegan: {e}, falling back to rules")
    
    # Fall back to rule-based classification (check each ingredient)
    return all(is_ingredient_vegan(ing) for ing in ingredients_list)