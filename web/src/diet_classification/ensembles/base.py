#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes and utilities for ensemble methods.

This module provides the foundation for all ensemble implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ..core import log


class BaseEnsemble(BaseEstimator, ClassifierMixin, ABC):
    """
    Abstract base class for all ensemble methods.
    
    Provides common functionality and interface for ensemble models.
    """
    
    def __init__(
        self,
        models: Dict[str, BaseEstimator],
        task: str,
        weights: Optional[np.ndarray] = None
    ):
        """
        Initialize base ensemble.
        
        Args:
            models: Dictionary of model_name -> model
            task: Task name ('keto' or 'vegan')
            weights: Optional weights for models
        """
        self.models = models
        self.task = task
        self.weights = weights
        self.n_models = len(models)
        
        # Validate weights
        if weights is not None:
            if len(weights) != self.n_models:
                raise ValueError(f"Weight length {len(weights)} != number of models {self.n_models}")
            
            # Normalize weights
            self.weights = np.array(weights) / np.sum(weights)
        else:
            # Equal weights by default
            self.weights = np.ones(self.n_models) / self.n_models
    
    def fit(self, X, y):
        """Ensemble models are already fitted."""
        return self
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Must be implemented by subclasses.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array of shape (n_samples, 2)
        """
        pass
    
    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def get_model_predictions(self, X) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions.
        
        Useful for analysis and debugging.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of model_name -> predictions
        """
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'decision_function'):
                    # Convert decision function to probability
                    scores = model.decision_function(X)
                    pred = 1 / (1 + np.exp(-scores))
                else:
                    # Binary predictions only
                    pred = model.predict(X).astype(float)
                
                predictions[name] = pred
                
            except Exception as e:
                log.warning(f"Model {name} prediction failed: {e}")
                predictions[name] = np.full(X.shape[0], 0.5)
        
        return predictions
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_models={self.n_models}, task={self.task})"


class EnsembleWrapper(BaseEnsemble):
    """
    Flexible wrapper for different ensemble strategies.
    
    Supports multiple ensemble types through a unified interface.
    """
    
    def __init__(
        self,
        ensemble_type: str,
        models: Union[Dict[str, BaseEstimator], BaseEstimator],
        weights: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        task: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ensemble wrapper.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'averaging', 'best_two', etc.)
            models: Dictionary of models or single ensemble model
            weights: Model weights
            alpha: Blending parameter for cross-domain ensembles
            task: Task name
            **kwargs: Additional parameters
        """
        # Handle different model inputs
        if isinstance(models, dict):
            super().__init__(models, task or "unknown", weights)
        else:
            # Pre-built ensemble (e.g., VotingClassifier)
            self.models = models
            self.task = task or "unknown"
            self.weights = weights
            self.n_models = getattr(models, 'n_estimators', len(getattr(models, 'estimators_', [])))
        
        self.ensemble_type = ensemble_type
        self.alpha = alpha
        self.kwargs = kwargs
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict probabilities based on ensemble type.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array
        """
        if self.ensemble_type == 'voting':
            # Delegate to sklearn VotingClassifier
            if hasattr(self.models, 'predict_proba'):
                return self.models.predict_proba(X)
            else:
                # Manual voting
                return self._voting_predict_proba(X)
        
        elif self.ensemble_type == 'averaging':
            return self._averaging_predict_proba(X)
        
        elif self.ensemble_type == 'best_two':
            return self._best_two_predict_proba(X)
        
        elif self.ensemble_type == 'stacking':
            return self._stacking_predict_proba(X)
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def _voting_predict_proba(self, X) -> np.ndarray:
        """Soft voting ensemble."""
        predictions = self.get_model_predictions(X)
        probs = np.array(list(predictions.values()))
        
        # Weighted average
        weighted_prob = np.average(probs, axis=0, weights=self.weights)
        
        return np.c_[1 - weighted_prob, weighted_prob]
    
    def _averaging_predict_proba(self, X) -> np.ndarray:
        """Simple averaging ensemble."""
        predictions = self.get_model_predictions(X)
        probs = np.array(list(predictions.values()))
        
        # Simple average
        avg_prob = np.mean(probs, axis=0)
        
        return np.c_[1 - avg_prob, avg_prob]
    
    def _best_two_predict_proba(self, X) -> np.ndarray:
        """Alpha blending of two models."""
        if 'text' not in self.models or 'image' not in self.models:
            raise ValueError("best_two requires 'text' and 'image' models")
        
        text_prob = self.models['text'].predict_proba(X)[:, 1]
        image_prob = self.models['image'].predict_proba(X)[:, 1]
        
        # Alpha blending
        blended = self.alpha * image_prob + (1 - self.alpha) * text_prob
        
        return np.c_[1 - blended, blended]
    
    def _stacking_predict_proba(self, X) -> np.ndarray:
        """Stacking ensemble (requires meta-learner)."""
        if 'meta_learner' not in self.kwargs:
            raise ValueError("Stacking requires meta_learner")
        
        # Get base model predictions
        predictions = self.get_model_predictions(X)
        meta_features = np.column_stack(list(predictions.values()))
        
        # Use meta-learner
        meta_learner = self.kwargs['meta_learner']
        
        if hasattr(meta_learner, 'predict_proba'):
            return meta_learner.predict_proba(meta_features)
        else:
            pred = meta_learner.predict(meta_features)
            return np.c_[1 - pred, pred]


def create_ensemble(
    ensemble_type: str,
    models: Dict[str, BaseEstimator],
    task: str,
    **kwargs
) -> BaseEnsemble:
    """
    Factory function to create ensemble instances.
    
    Args:
        ensemble_type: Type of ensemble
        models: Dictionary of models
        task: Task name
        **kwargs: Additional parameters
        
    Returns:
        Ensemble instance
    """
    ensemble_type = ensemble_type.lower()
    
    if ensemble_type in ['voting', 'averaging', 'best_two', 'stacking']:
        return EnsembleWrapper(
            ensemble_type=ensemble_type,
            models=models,
            task=task,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def evaluate_ensemble_diversity(
    models: Dict[str, BaseEstimator],
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate diversity among ensemble members.
    
    Diversity metrics help understand if models make different errors.
    
    Args:
        models: Dictionary of models
        X: Feature matrix
        y: True labels
        
    Returns:
        Dictionary of diversity metrics
    """
    predictions = {}
    
    # Get predictions from all models
    for name, model in models.items():
        pred = model.predict(X)
        predictions[name] = pred
    
    # Convert to array
    pred_matrix = np.array(list(predictions.values()))
    n_models = pred_matrix.shape[0]
    
    # Calculate pairwise disagreement
    disagreements = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagreement = np.mean(pred_matrix[i] != pred_matrix[j])
            disagreements.append(disagreement)
    
    # Calculate Q-statistic (Yule's Q)
    q_statistics = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Contingency table
            n11 = np.sum((pred_matrix[i] == 1) & (pred_matrix[j] == 1))
            n10 = np.sum((pred_matrix[i] == 1) & (pred_matrix[j] == 0))
            n01 = np.sum((pred_matrix[i] == 0) & (pred_matrix[j] == 1))
            n00 = np.sum((pred_matrix[i] == 0) & (pred_matrix[j] == 0))
            
            # Q-statistic
            numerator = n11 * n00 - n01 * n10
            denominator = n11 * n00 + n01 * n10
            
            if denominator > 0:
                q = numerator / denominator
                q_statistics.append(q)
    
    # Entropy measure
    # For each sample, count how many models predict class 1
    votes = np.sum(pred_matrix, axis=0)
    # Normalize to [0, 1]
    vote_proportions = votes / n_models
    # Calculate entropy
    entropy = -vote_proportions * np.log2(vote_proportions + 1e-10) - \
              (1 - vote_proportions) * np.log2(1 - vote_proportions + 1e-10)
    avg_entropy = np.mean(entropy)
    
    return {
        'avg_disagreement': np.mean(disagreements) if disagreements else 0,
        'avg_q_statistic': np.mean(q_statistics) if q_statistics else 0,
        'avg_entropy': avg_entropy,
        'n_models': n_models
    }