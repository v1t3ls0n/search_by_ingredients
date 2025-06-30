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


def is_vegan(ingredients: Union[Iterable[str], str], mode: Optional[str] = 'text') -> bool:
    """
    Check if all ingredients in a recipe are vegan.

    Uses the best available method for the specified mode:
    1. Mode-specific trained ML model (if available)
    2. General best model (if no mode-specific model)
    3. Falls back to rule-based classification per ingredient

    Args:
        ingredients: Either a string (comma-separated or JSON) or an iterable
        mode: Optional mode ('text', 'image', or 'both') to use specific model

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
    
    # Determine the best model to use based on mode
    model = None
    model_source = None
    
    if mode:
        # Try to get mode-specific model first
        from ..models.io import load_best_model_for_mode
        model = load_best_model_for_mode('vegan', mode)
        if model:
            model_source = f"{mode}-specific"
            log.debug(f"Using {mode}-specific model for vegan classification")
    
    # If no mode specified or mode-specific model not found, try to determine best model
    if model is None and pipeline_state.vectorizer:
        # Check what models are available in pipeline state
        available_models = []
        
        # Check for domain-specific models
        for domain in ['ensemble', 'both', 'text', 'image']:
            key = f"vegan_{domain}"
            if key in pipeline_state.models:
                available_models.append((domain, pipeline_state.models[key]))
        
        # Also check for simple task key
        if 'vegan' in pipeline_state.models:
            available_models.append(('general', pipeline_state.models['vegan']))
        
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
                    log.debug(f"Using {avail_domain} model for vegan classification")
                    break
            if model:
                break
    
    # Final fallback to general best model
    if model is None:
        model = pipeline_state.get_best_model('vegan')
        if model:
            model_source = "general best"
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
            prob_adj = verify_with_rules("vegan", pd.Series([normalized]), np.array([prob]))[0]
            
            log.debug(f"Vegan prediction using {model_source} model: {prob_adj:.3f}")
            
            return prob_adj >= 0.5
            
        except Exception as e:
            log.debug(f"ML prediction failed for vegan using {model_source} model: {e}, falling back to rules")
    
    # Fall back to rule-based classification (check each ingredient)
    log.debug("Using rule-based classification for vegan")
    return all(is_ingredient_vegan(ing) for ing in ingredients_list)