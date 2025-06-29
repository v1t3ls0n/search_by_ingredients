#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based classification model.

Based on original lines 2797-2863 from diet_classifiers.py
"""

import numpy as np

# Handle sklearn availability
try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback base classes if sklearn not available
    class BaseEstimator:
        pass
    
    class ClassifierMixin:
        pass
    
    SKLEARN_AVAILABLE = False


class RuleModel(BaseEstimator, ClassifierMixin):
    """
    Rule-based classifier for diet classification.

    Implements the scikit-learn interface while using only regex patterns
    and domain rules for classification. Provides probabilistic outputs
    with high confidence for rule matches.

    Attributes:
        task: Classification task ('keto' or 'vegan')
        rx_black: Compiled regex for blacklist patterns
        rx_white: Compiled regex for whitelist patterns
        pos_prob: Probability assigned to positive class (default 0.98)
        neg_prob: Probability assigned to negative class (default 0.02)
        
    Based on original lines 2797-2863
    """

    def __init__(self, task: str, rx_black=None, rx_white=None,
                 pos_prob=0.98, neg_prob=0.02):
        """
        Initialize rule-based model.

        Args:
            task: 'keto' or 'vegan'
            rx_black: Regex pattern for blacklisted items
            rx_white: Regex pattern for whitelisted items
            pos_prob: Confidence for positive predictions
            neg_prob: Confidence for negative predictions
        """
        self.task = task
        self.rx_black = rx_black
        self.rx_white = rx_white
        self.pos_prob = pos_prob
        self.neg_prob = neg_prob

    def fit(self, X, y=None):
        """No training needed for rule-based model."""
        return self

    def _pos(self, d: str) -> bool:
        """
        Determine if ingredient is positive class using full rule pipeline.

        Delegates to the main classification functions to ensure
        consistency with the overall system.
        """
        if self.task == "keto":
            from ..classification.keto import is_ingredient_keto
            return is_ingredient_keto(d)
        else:  # vegan
            from ..classification.vegan import is_ingredient_vegan
            return is_ingredient_vegan(d)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples.

        Args:
            X: Array-like of ingredient strings

        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        p = np.fromiter(
            (self.pos_prob if self._pos(d) else self.neg_prob for d in X),
            float,
            count=len(X)
        )
        return np.c_[1 - p, p]

    def predict(self, X):
        """
        Predict class labels for samples.

        Args:
            X: Array-like of ingredient strings

        Returns:
            Array of binary predictions (0 or 1)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)