#!/usr/bin/env python3

"""
Compatibility wrapper for diet classification system - exposes only public API.

This file maintains backward compatibility with existing code that imports
from diet_classifiers while the actual implementation is refactored into
the diet_classification package.
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

# Re-export ONLY the required public APIs
from diet_classification.classification.keto import is_keto, is_ingredient_keto
from diet_classification.classification.vegan import is_vegan, is_ingredient_vegan

# For CLI compatibility - when running as script
if __name__ == "__main__":
    from diet_classification.__main__ import main
    main()