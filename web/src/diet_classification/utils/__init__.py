#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility modules for diet classification.
"""

from .constants import (
    NON_KETO, NON_VEGAN, 
    KETO_WHITELIST, VEGAN_WHITELIST,
    get_keto_patterns, get_vegan_patterns
)
from .memory import (
    get_available_memory, 
    optimize_memory_usage, 
    handle_memory_crisis
)
from .validation import preflight_checks

__all__ = [
    # Constants
    'NON_KETO', 'NON_VEGAN',
    'KETO_WHITELIST', 'VEGAN_WHITELIST',
    'get_keto_patterns', 'get_vegan_patterns',
    
    # Memory management
    'get_available_memory',
    'optimize_memory_usage',
    'handle_memory_crisis',
    
    # Validation
    'preflight_checks',
]