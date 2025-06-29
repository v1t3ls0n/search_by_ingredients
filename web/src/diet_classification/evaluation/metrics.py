#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics calculation and display utilities.

Based on original lines 5223-5238, 5273-5284 from diet_classifiers.py
"""

import numpy as np

from ..core import log

# Handle sklearn availability
try:
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, precision_score, 
        recall_score, f1_score, roc_auc_score, average_precision_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not available - metrics functionality limited")


def table(title, rows):
    """
    Display results in a formatted table.

    Args:
        title: Table title
        rows: List of result dictionaries
        
    Based on original lines 5223-5238
    """
    cols = ("ACC", "PREC", "REC", "F1", "ROC", "PR")
    pad = 11 + 8 * len(cols)
    hdr = "│ model task " + " ".join(f"{c:>7}" for c in cols) + " │"
    log.info(f"\n╭─ {title} {'─' * (pad - len(title) - 2)}")
    log.info(hdr)
    log.info("├" + "─" * (len(hdr) - 2) + "┤")
    for r in rows:
        vals = " ".join(f"{r[c]:>7.2f}" for c in cols)
        log.info(f"│ {r['model']:<7} {r['task']:<5} {vals} │")
    log.info("╰" + "─" * (len(hdr) - 2) + "╯")


def pack(y, prob):
    """
    Calculate comprehensive metrics from predictions.

    Args:
        y: True labels
        prob: Predicted probabilities

    Returns:
        Dictionary of metrics
        
    Based on original lines 5273-5284
    """
    if not SKLEARN_AVAILABLE:
        log.error("scikit-learn required for metrics calculation")
        return {
            "ACC": 0.0,
            "PREC": 0.0,
            "REC": 0.0,
            "F1": 0.0,
            "ROC": 0.0,
            "PR": 0.0
        }
    
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        ACC=accuracy_score(y, pred),
        PREC=precision_score(y, pred, zero_division=0),
        REC=recall_score(y, pred, zero_division=0),
        F1=f1_score(y, pred, zero_division=0),
        ROC=roc_auc_score(y, prob),
        PR=average_precision_score(y, prob)
    )


def ensure_predict_proba(estimator, X_train, y_train):
    """
    Ensure estimator has predict_proba method by wrapping with calibration if needed.

    Some models (like SVC) don't provide probability estimates by default.
    This function wraps them with CalibratedClassifierCV to enable probabilities.

    Args:
        estimator: Fitted estimator
        X_train: Training features
        y_train: Training labels

    Returns:
        Estimator with predict_proba capability
        
    Based on original lines 4108-4130
    """
    if not hasattr(estimator, "predict_proba"):
        log.info(
            f"Adding probability calibration to {estimator.__class__.__name__}")
        try:
            from sklearn.calibration import CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(
                estimator, cv=3, method='sigmoid')
            calibrated.fit(X_train, y_train)
            return calibrated
        except Exception as e:
            log.error(f"Calibration failed: {e}")
            return estimator
    return estimator