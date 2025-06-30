#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keto diet classification functions.

Based on original lines 2159-2350, 1965-1989 from diet_classifiers.py
"""

import json
from typing import Iterable, Union, Optional
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


def is_keto(ingredients: Union[Iterable[str], str], mode: Optional[str] = 'text') -> bool:
    """
    Check if all ingredients in a recipe are keto-friendly.

    Uses the best available method for the specified mode:
    1. Mode-specific trained ML model (if available)
    2. General best model (if no mode-specific model)
    3. Falls back to rule-based classification per ingredient

    Args:
        ingredients: Either a string (comma-separated or JSON) or an iterable
        mode: Optional mode ('text', 'image', or 'both') to use specific model

    Returns:
        True if keto-friendly, False otherwise

    Example:
        >>> is_keto("almond flour, eggs, butter")
        True
        >>> is_keto(["chicken", "broccoli", "rice"])
        False  # Rice is not keto
        >>> is_keto('["spinach", "cheese", "cream"]')
        True
        >>> is_keto("almond flour, eggs", mode="text")  # Use text-only model
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
    
    # Determine the best model to use based on mode
    model = None
    model_source = None
    
    if mode:
        # Try to get mode-specific model first
        from ..models.io import load_best_model_for_mode
        model = load_best_model_for_mode('keto', mode)
        if model:
            model_source = f"{mode}-specific"
            log.debug(f"Using {mode}-specific model for keto classification")
    
    # If no mode specified or mode-specific model not found, try to determine best model
    if model is None and pipeline_state.vectorizer:
        # Check what models are available in pipeline state
        available_models = []
        
        # Check for domain-specific models
        for domain in ['ensemble', 'both', 'text', 'image']:
            key = f"keto_{domain}"
            if key in pipeline_state.models:
                available_models.append((domain, pipeline_state.models[key]))
        
        # Also check for simple task key
        if 'keto' in pipeline_state.models:
            available_models.append(('general', pipeline_state.models['keto']))
        
        # Select best model based on mode preference
        if mode == 'text':
            # Prefer text model, then ensemble, then both
            preference_order = ['text', 'ensemble', 'both', 'general']
        elif mode == 'image':
            # Prefer image model, then ensemble, then both
            preference_order = ['image', 'ensemble', 'both', 'general']
        elif mode == 'both':
            # Prefer both, then ensemble
            preference_order = ['both', 'ensemble', 'text', 'image', 'general']
        else:
            # No mode specified - use best available
            preference_order = ['ensemble', 'both', 'text', 'image', 'general']
        
        # Find first available model in preference order
        for pref_domain in preference_order:
            for avail_domain, avail_model in available_models:
                if avail_domain == pref_domain:
                    model = avail_model
                    model_source = avail_domain
                    log.debug(f"Using {avail_domain} model for keto classification")
                    break
            if model:
                break
    
    # Final fallback to general best model
    if model is None:
        model = pipeline_state.get_best_model('keto')
        if model:
            model_source = "general best"
            log.debug(f"Using general best model for keto classification")
    
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
                    # Model expects images
                    if mode == 'image':
                        log.warning("Image model requested but no image features available")
                        # Fall back to rules
                        return all(is_ingredient_keto(ing) for ing in ingredients_list)
                    elif mode == 'both' or (mode is None and expected_features > actual_features + 2048):
                        # Pad with zeros for missing image features
                        from ..features.combiners import combine_features
                        padding = np.zeros((1, 2048), dtype=np.float32)
                        X = combine_features(X, padding)
                        log.debug(f"Padded features for {model_source} model expecting images")
                elif expected_features < actual_features and mode == 'image':
                    # Text model but image mode requested
                    log.warning(f"Using {model_source} model but image mode was requested")
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[0, 1]
            else:
                prob = float(model.predict(X)[0])
            
            # Apply rule-based verification
            from ..classification.verification import verify_with_rules
            prob_adj = verify_with_rules("keto", pd.Series([norm]), np.array([prob]))[0]
            
            log.debug(f"Keto prediction using {model_source} model: {prob_adj:.3f}")
            
            return prob_adj >= 0.5
            
        except Exception as e:
            log.debug(f"ML prediction failed for keto using {model_source} model: {e}, falling back to rules")
    
    # Fall back to rule-based classification (check each ingredient)
    log.debug("Using rule-based classification for keto")
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