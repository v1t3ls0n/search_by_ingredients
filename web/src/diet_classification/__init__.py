#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diet Classification Package - Multi-modal ML for ingredient analysis.

This package provides machine learning models for classifying recipes as
keto-friendly or vegan based on their ingredients using text and image features.
"""

__version__ = "1.0.0"

# Only export the public API functions
from .classification.keto import is_keto, is_ingredient_keto
from .classification.vegan import is_vegan, is_ingredient_vegan

__all__ = [
    'is_keto',
    'is_ingredient_keto',
    'is_vegan',
    'is_ingredient_vegan',
]