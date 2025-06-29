#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diet classification functions.
"""

from .keto import is_keto, is_ingredient_keto, find_non_keto_hits
from .vegan import is_vegan, is_ingredient_vegan
from .verification import verify_with_rules

__all__ = [
    # Keto classification
    'is_keto',
    'is_ingredient_keto',
    'find_non_keto_hits',
    
    # Vegan classification
    'is_vegan',
    'is_ingredient_vegan',
    
    # Verification
    'verify_with_rules',
]