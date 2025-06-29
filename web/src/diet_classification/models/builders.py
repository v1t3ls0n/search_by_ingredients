#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model building functions for different feature domains.

Based on original lines 3877-4033 from diet_classifiers.py
"""

import os
from typing import Dict, Any

from ..core import log
from ..utils.memory import get_available_memory

# Handle sklearn availability
try:
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import (
        LogisticRegression, SGDClassifier,
        PassiveAggressiveClassifier, RidgeClassifier
    )
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = object

# Handle LightGBM availability
try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def build_models(task: str, domain: str = "text") -> Dict[str, BaseEstimator]:
    """
    Build a dictionary of ML models appropriate for the given domain.

    Enhanced with resource-aware model selection that adapts to available memory.

    Different models are optimized for different feature types:
    - Text models: Naive Bayes, linear models (work well with sparse TF-IDF)
    - Image models: Neural networks, tree-based models (handle dense features)
    - Both: All models available

    Args:
        task: Classification task ('keto' or 'vegan')
        domain: Feature domain ('text', 'image', or 'both')

    Returns:
        Dictionary mapping model names to estimator instances
        
    Based on original lines 3877-4033
    """
    models: Dict[str, BaseEstimator] = {}

    if not SKLEARN_AVAILABLE:
        log.error("scikit-learn not available - cannot build models")
        return models

    # Check available resources
    available_memory = get_available_memory()
    is_sanity_check = os.environ.get('SANITY_CHECK') == '1'

    log.info(f"   🔧 Building models for {task} ({domain} domain)")
    log.info(f"   ├─ Available memory: {available_memory:.1f} GB")
    log.info(f"   └─ Sanity check mode: {is_sanity_check}")

    # If in sanity check mode, return minimal models
    if is_sanity_check:
        log.info(f"   🏃 Sanity check mode - using minimal model set")
        if domain == "text":
            return {
                "NB": MultinomialNB(),
                "Ridge": RidgeClassifier(class_weight="balanced", random_state=42)
            }
        else:
            return {
                "Softmax": LogisticRegression(
                    solver="lbfgs", max_iter=100, random_state=42
                )
            }

    # Text-oriented models (work well with sparse features)
    text_family: Dict[str, BaseEstimator] = {
        "NB": MultinomialNB(),  # Classic for text classification
        "Softmax": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge": RidgeClassifier(class_weight="balanced", random_state=42),
        "PA": PassiveAggressiveClassifier(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "SGD": SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "LinearSVC": LinearSVC(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            dual=False,  # Use primal optimization for large datasets
            tol=1e-3
        ),
    }

    # Image/mixed-feature models (handle dense features better)
    image_family: Dict[str, BaseEstimator] = {}

    # Adapt models based on available memory
    if available_memory >= 16:  # High memory - use all models
        log.info(f"   💪 High memory mode - all models available")

        image_family["RF"] = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

        if lgb:
            image_family["LGBM"] = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                objective="binary",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True,
            )

    elif available_memory >= 8:  # Medium memory - reduce model complexity
        log.info(f"   ⚡ Medium memory mode - reduced model complexity")

        image_family["RF"] = RandomForestClassifier(
            n_estimators=100,  # Reduced from 150
            max_depth=15,      # Reduced from 20
            min_samples_split=10,  # Increased from 5
            min_samples_leaf=5,    # Increased from 2
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

        if lgb:
            image_family["LGBM"] = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=50,   # Reduced from 100
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                objective="binary",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True,
                # Memory-saving parameters
                max_bin=127,       # Reduced from default 255
                min_data_in_bin=10,
            )

    else:  # Low memory - minimal models only
        log.warning(
            f"   ⚠️  Low memory mode ({available_memory:.1f} GB) - using minimal models")

        # Only lightweight model for image features
        image_family["RF"] = RandomForestClassifier(
            n_estimators=50,   # Minimal trees
            max_depth=10,      # Shallow trees
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=2,  # Limit parallelism
            random_state=42,
        )

        # Remove memory-intensive text models
        text_family.pop("LinearSVC", None)
        text_family.pop("SGD", None)

    # Assemble model selection based on domain
    if domain == "text":
        models.update(text_family)
    elif domain == "image":
        models.update(image_family)
    elif domain == "both":
        models.update(text_family)
        models.update(image_family)

    # Log final model selection
    log.info(f"   📦 Selected models: {list(models.keys())}")

    return models


# Hyperparameter grids for tuning
HYPER = {
    # Text models
    "Softmax": {
        "C": [0.1, 1, 10]  # Regularization strength
    },
    "SGD": {
        "alpha": [1e-4, 1e-3],  # Regularization
        "loss": ["log_loss", "modified_huber"],  # Loss function
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0]  # Regularization
    },
    "PA": {
        "C": [0.5, 1.0]  # Aggressiveness parameter
    },
    "NB": {},  # No hyperparameters to tune

    # Image/mixed models
    "MLP": {
        "hidden_layer_sizes": [(256,), (512, 128)],
        "alpha": [0.0001, 0.001],  # L2 regularization
        "learning_rate_init": [0.001, 0.005],
    },
    "RF": {
        "n_estimators": [150, 300],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
    },
    "LGBM": {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "n_estimators": [150, 250],
        "min_child_samples": [10, 20],
    },
    # For CalibratedSVM, parameters live under 'estimator__'
    "SVM_RBF": {
        "estimator__svc__C": [0.5, 1, 2],
        "estimator__svc__gamma": ["scale", 0.001]
    },
}

# Fast mode settings for development
FAST = True
CV = 2 if FAST else 3  # Cross-validation folds
N_IT = 2 if FAST else 6  # Number of iterations for random search