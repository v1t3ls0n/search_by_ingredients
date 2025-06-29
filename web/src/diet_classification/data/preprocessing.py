#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text preprocessing utilities for ingredient normalization.

Based on original lines 1944-2039 from diet_classifiers.py
"""

import re
import unicodedata
from typing import Union, List, Tuple
import numpy as np

try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    # Download required NLTK data
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        _LEMM = WordNetLemmatizer()
    except:
        _LEMM = None
except ImportError:
    _LEMM = None

# Pattern for common cooking units
_UNITS = re.compile(r"\b(?:g|gram|kg|oz|ml|l|cup|cups|tsp|tbsp|teaspoon|"
                    r"tablespoon|pound|lb|slice|slices|small|large|medium)\b")


def tokenize_ingredient(text: str) -> List[str]:
    """
    Extract word tokens from ingredient text.

    Splits text into individual words, handling hyphenated words correctly.
    Used for token-level matching in classification rules.

    Args:
        text: Raw ingredient text

    Returns:
        List of lowercase word tokens

    Example:
        >>> tokenize_ingredient("Sugar-free dark chocolate")
        ['sugar-free', 'dark', 'chocolate']
        
    Based on original lines 1944-1963
    """
    return re.findall(r"\b\w[\w-]*\b", text.lower())


def normalise(t: Union[str, list, tuple, np.ndarray]) -> str:
    """
    Normalize ingredient text for consistent matching.

    This comprehensive normalization function handles various input formats
    and applies multiple cleaning steps to prepare text for classification:

    1. Handles list/array inputs (from parquet files)
    2. Removes accents and non-ASCII characters
    3. Removes parenthetical information
    4. Removes cooking units and measurements
    5. Removes numbers and fractions
    6. Removes punctuation
    7. Applies lemmatization (if available)
    8. Filters out very short words

    Args:
        t: Input text (string, list, tuple, or numpy array)

    Returns:
        Normalized string with cleaned, lemmatized tokens

    Example:
        >>> normalise("2 cups of sliced bananas (ripe)")
        "slice banana ripe"
        >>> normalise(["olive oil", "salt"])
        "olive oil salt"
        
    Based on original lines 1973-2039
    """
    # Handle non-string inputs
    if not isinstance(t, str):
        if isinstance(t, (list, tuple, np.ndarray)):
            t = " ".join(map(str, t))
        else:
            t = str(t)

    # Unicode normalization - remove accents
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()

    # Remove parenthetical content and convert to lowercase
    t = re.sub(r"\([^)]*\)", " ", t.lower())

    # Remove units of measurement
    t = _UNITS.sub(" ", t)

    # Remove numbers (including fractions)
    t = re.sub(r"\d+(?:[/\.]\d+)?", " ", t)

    # Remove punctuation
    t = re.sub(r"[^\w\s-]", " ", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Apply lemmatization and filter short words
    if _LEMM:
        return " ".join(_LEMM.lemmatize(w) for w in t.split() if len(w) > 2)
    return " ".join(w for w in t.split() if len(w) > 2)