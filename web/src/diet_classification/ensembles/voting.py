#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voting ensemble implementations.

Provides both hard and soft voting strategies with various weighting schemes.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier

from ..core import log
from ..evaluation.metrics import pack
from .base import BaseEnsemble, EnsembleWrapper


def create_voting_ensemble(
    models: Dict[str, BaseEstimator],
    task: str,
    voting: str = 'soft',
    weights: Optional[List[float]] = None,
    flatten_transform: bool = True
) -> EnsembleWrapper:
    """
    Create a voting ensemble using sklearn's VotingClassifier.
    
    Args:
        models: Dictionary of model_name -> model
        task: Task name ('keto' or 'vegan')
        voting: 'hard' or 'soft' voting
        weights: Optional weights for models
        flatten_transform: Whether to flatten transform output
        
    Returns:
        EnsembleWrapper containing VotingClassifier
    """
    # Convert dict to list of tuples for VotingClassifier
    estimators = list(models.items())
    
    # Create VotingClassifier
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        flatten_transform=flatten_transform,
        n_jobs=-1
    )
    
    # Note: VotingClassifier needs to be fitted, but our models are already fitted
    # So we'll use the wrapper to handle predictions directly
    
    log.info(f"Created {voting} voting ensemble with {len(models)} models for {task}")
    
    return EnsembleWrapper(
        ensemble_type='voting',
        models=models,  # Pass original dict for direct access
        weights=weights,
        task=task,
        voting_type=voting
    )


def dynamic_weighted_voting(
    models: Dict[str, BaseEstimator],
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    metric: str = 'f1'
) -> Tuple[EnsembleWrapper, np.ndarray]:
    """
    Create voting ensemble with weights based on validation performance.
    
    Args:
        models: Dictionary of models
        X_val: Validation features
        y_val: Validation labels
        task: Task name
        metric: Metric to use for weighting ('f1', 'accuracy', 'precision', 'recall')
        
    Returns:
        Tuple of (ensemble, weights)
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    
    metric_funcs = {
        'f1': f1_score,
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)
    }
    
    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Calculate performance for each model
    performances = {}
    metric_func = metric_funcs[metric]
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_val)
            score = metric_func(y_val, y_pred)
            performances[name] = score
            log.info(f"   {name}: {metric}={score:.3f}")
        except Exception as e:
            log.warning(f"   {name}: Failed to evaluate - {e}")
            performances[name] = 0.0
    
    # Convert scores to weights
    scores = np.array(list(performances.values()))
    
    # Handle edge cases
    if np.sum(scores) == 0:
        # All models failed - use equal weights
        weights = np.ones(len(models)) / len(models)
    else:
        # Normalize scores to weights
        weights = scores / np.sum(scores)
    
    log.info(f"Dynamic weights based on {metric}: {dict(zip(performances.keys(), weights))}")
    
    # Create ensemble
    ensemble = create_voting_ensemble(
        models=models,
        task=task,
        voting='soft',
        weights=weights.tolist()
    )
    
    return ensemble, weights


def threshold_voting_ensemble(
    models: Dict[str, BaseEstimator],
    task: str,
    confidence_threshold: float = 0.7,
    min_votes: int = None
) -> EnsembleWrapper:
    """
    Create ensemble that only votes when models are confident.
    
    Models only contribute to the vote if their prediction confidence
    exceeds the threshold.
    
    Args:
        models: Dictionary of models
        task: Task name
        confidence_threshold: Minimum confidence to participate in vote
        min_votes: Minimum number of votes required (defaults to majority)
        
    Returns:
        Custom voting ensemble
    """
    if min_votes is None:
        min_votes = len(models) // 2 + 1
    
    class ThresholdVotingEnsemble(BaseEnsemble):
        def __init__(self, models, task, threshold, min_votes):
            super().__init__(models, task)
            self.threshold = threshold
            self.min_votes = min_votes
        
        def predict_proba(self, X):
            # Get predictions from all models
            all_probs = []
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)[:, 1]
                else:
                    # Use binary predictions
                    probs = model.predict(X).astype(float)
                
                all_probs.append(probs)
            
            all_probs = np.array(all_probs)  # Shape: (n_models, n_samples)
            
            # For each sample, vote only if confident
            final_probs = []
            
            for i in range(X.shape[0]):
                sample_probs = all_probs[:, i]
                
                # Find confident predictions
                confident_pos = sample_probs >= self.threshold
                confident_neg = sample_probs <= (1 - self.threshold)
                confident_mask = confident_pos | confident_neg
                
                n_confident = np.sum(confident_mask)
                
                if n_confident >= self.min_votes:
                    # Use only confident predictions
                    confident_probs = sample_probs[confident_mask]
                    avg_prob = np.mean(confident_probs)
                else:
                    # Not enough confident predictions - use all
                    avg_prob = np.mean(sample_probs)
                
                final_probs.append(avg_prob)
            
            final_probs = np.array(final_probs)
            return np.c_[1 - final_probs, final_probs]
    
    ensemble = ThresholdVotingEnsemble(models, task, confidence_threshold, min_votes)
    
    # Wrap in EnsembleWrapper for consistency
    return EnsembleWrapper(
        ensemble_type='voting',
        models=ensemble,
        task=task,
        voting_type='threshold',
        confidence_threshold=confidence_threshold,
        min_votes=min_votes
    )


def ranked_voting_ensemble(
    models: Dict[str, BaseEstimator],
    model_rankings: Dict[str, int],
    task: str,
    rank_weight_func: callable = None
) -> EnsembleWrapper:
    """
    Create voting ensemble where votes are weighted by model rankings.
    
    Args:
        models: Dictionary of models
        model_rankings: Dictionary of model_name -> rank (1 is best)
        task: Task name
        rank_weight_func: Function to convert rank to weight (default: 1/rank)
        
    Returns:
        Weighted voting ensemble
    """
    if rank_weight_func is None:
        rank_weight_func = lambda rank: 1.0 / rank
    
    # Calculate weights from rankings
    weights = []
    for name, model in models.items():
        rank = model_rankings.get(name, len(models))  # Default to worst rank
        weight = rank_weight_func(rank)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    log.info(f"Rank-based weights: {dict(zip(models.keys(), weights))}")
    
    return create_voting_ensemble(
        models=models,
        task=task,
        voting='soft',
        weights=weights.tolist()
    )


def calibrated_voting_ensemble(
    models: Dict[str, BaseEstimator],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    task: str,
    method: str = 'isotonic'
) -> EnsembleWrapper:
    """
    Create voting ensemble with calibrated probabilities.
    
    Uses probability calibration to improve ensemble predictions.
    
    Args:
        models: Dictionary of models
        X_cal: Calibration features
        y_cal: Calibration labels
        task: Task name
        method: Calibration method ('isotonic' or 'sigmoid')
        
    Returns:
        Calibrated voting ensemble
    """
    from sklearn.calibration import CalibratedClassifierCV
    
    # Calibrate each model
    calibrated_models = {}
    
    for name, model in models.items():
        try:
            # Check if model supports probability prediction
            if hasattr(model, 'predict_proba'):
                # Calibrate using validation data
                calibrated = CalibratedClassifierCV(
                    model,
                    method=method,
                    cv='prefit'  # Model is already fitted
                )
                calibrated.fit(X_cal, y_cal)
                calibrated_models[name] = calibrated
                log.info(f"   Calibrated {name} using {method}")
            else:
                # Keep original model
                calibrated_models[name] = model
                log.info(f"   {name} does not support probabilities - keeping original")
                
        except Exception as e:
            log.warning(f"   Failed to calibrate {name}: {e}")
            calibrated_models[name] = model
    
    # Create voting ensemble with calibrated models
    return create_voting_ensemble(
        models=calibrated_models,
        task=task,
        voting='soft'
    )


def hierarchical_voting_ensemble(
    models_by_type: Dict[str, Dict[str, BaseEstimator]],
    task: str,
    type_weights: Optional[Dict[str, float]] = None
) -> EnsembleWrapper:
    """
    Create hierarchical voting ensemble.
    
    First aggregates predictions within each model type, then combines
    type-level predictions.
    
    Args:
        models_by_type: Dictionary of type -> {model_name -> model}
        task: Task name
        type_weights: Optional weights for each type
        
    Returns:
        Hierarchical voting ensemble
    """
    class HierarchicalVotingEnsemble(BaseEnsemble):
        def __init__(self, models_by_type, task, type_weights):
            # Flatten all models for base class
            all_models = {}
            for type_models in models_by_type.values():
                all_models.update(type_models)
            
            super().__init__(all_models, task)
            
            self.models_by_type = models_by_type
            self.type_weights = type_weights or {
                t: 1.0/len(models_by_type) for t in models_by_type
            }
        
        def predict_proba(self, X):
            type_predictions = {}
            
            # Get predictions for each type
            for model_type, type_models in self.models_by_type.items():
                type_probs = []
                
                for name, model in type_models.items():
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X)[:, 1]
                    else:
                        prob = model.predict(X).astype(float)
                    
                    type_probs.append(prob)
                
                # Average within type
                type_avg = np.mean(type_probs, axis=0)
                type_predictions[model_type] = type_avg
            
            # Weighted average across types
            final_prob = np.zeros(X.shape[0])
            
            for model_type, type_prob in type_predictions.items():
                weight = self.type_weights.get(model_type, 0)
                final_prob += weight * type_prob
            
            return np.c_[1 - final_prob, final_prob]
    
    ensemble = HierarchicalVotingEnsemble(models_by_type, task, type_weights)
    
    return EnsembleWrapper(
        ensemble_type='voting',
        models=ensemble,
        task=task,
        voting_type='hierarchical'
    )