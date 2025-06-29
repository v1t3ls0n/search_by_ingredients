#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based verification for ML predictions.

Based on original lines 2869-2919 from diet_classifiers.py
"""

import numpy as np
import pandas as pd

from ..core import log
from ..utils.constants import get_keto_patterns, get_vegan_patterns
from ..data.preprocessing import normalise, tokenize_ingredient
from ..data.silver_labels import is_keto_ingredient_list


def verify_with_rules(task: str, clean: pd.Series, prob: np.ndarray) -> np.ndarray:
    """
    Apply rule-based verification to ML predictions.

    This function ensures that domain rules always take precedence over
    ML predictions. It's a critical safety layer that prevents the model
    from making obvious mistakes.

    Args:
        task: 'keto' or 'vegan'
        clean: Series of normalized ingredient texts
        prob: Array of ML predicted probabilities

    Returns:
        Adjusted probability array with rule corrections applied
        
    Based on original lines 2869-2919
    """
    adjusted = prob.copy()

    if task == "keto":
        patterns = get_keto_patterns()
        
        # Regex-based whitelist/blacklist
        is_whitelisted = clean.str.contains(patterns['whitelist'], na=False)
        is_blacklisted = clean.str.contains(patterns['blacklist'], na=False)
        forced_non_keto = is_blacklisted & ~is_whitelisted
        adjusted[forced_non_keto.values] = 0.0

        # Token-based ingredient verification
        for i, txt in enumerate(clean):
            if adjusted[i] > 0.5:
                tokens = tokenize_ingredient(normalise(txt))
                if not is_keto_ingredient_list(tokens):
                    adjusted[i] = 0.0
                    log.debug("Heuristically rejected '%s' as non-keto", txt)

        if forced_non_keto.any():
            log.debug("Keto Verification: forced %d probs to 0 (regex)",
                      forced_non_keto.sum())

    else:  # vegan
        patterns = get_vegan_patterns()
        
        bad = clean.str.contains(patterns['blacklist'], na=False) & ~clean.str.contains(patterns['whitelist'], na=False)
        adjusted[bad.values] = 0.0
        if bad.any():
            log.debug("Vegan Verification: forced %d probs to 0", bad.sum())

    return adjusted