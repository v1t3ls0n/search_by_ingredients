#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blending ensemble implementations.

Provides various blending strategies for combining model predictions.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from ..core import log
from ..evaluation.metrics import pack
from ..classification.verification import verify_with_rules
from .base import BaseEnsemble, EnsembleWrapper


def best_two_domains(
    task: str,
    text_results: List[Dict],
    image_results: List[Dict],
    gold_df: pd.DataFrame,
    alpha: float = 0.5
) -> Dict:
    """
    Blend predictions from best text and image models.
    
    This is a cross-domain blending strategy that combines the strengths
    of different feature modalities.
    
    Args:
        task: Task name ('keto' or 'vegan')
        text_results: List of text model results
        image_results: List of image model results
        gold_df: Gold standard DataFrame
        alpha: Blending weight (alpha * image + (1-alpha) * text)
        
    Returns:
        Result dictionary with blended predictions
    """
    # Find best models from each domain
    best_text = max(
        (r for r in text_results if r["task"] == task),
        key=lambda r: r["F1"]
    )
    
    best_img = max(
        (r for r in image_results if r["task"] == task and len(r.get("prob", [])) > 0),
        key=lambda r: r["F1"],
        default=None
    )
    
    log.info(f"🔀 BEST-TWO BLENDING ({task})")
    log.info(f"   ├─ Text: {best_text['model']} (F1={best_text['F1']:.3f})")
    log.info(f"   ├─ Image: {best_img['model'] if best_img else 'N/A'}")
    log.info(f"   └─ Alpha: {alpha}")
    
    # Get predictions
    txt_prob = best_text["prob"]
    y_true = gold_df[f"label_{task}"].values
    
    if best_img is None:
        log.warning("   ⚠️  No image model available - using text only")
        final_prob = txt_prob
    else:
        # Handle different sample sizes
        img_prob = best_img["prob"]
        img_indices = gold_df.index[:len(img_prob)]
        
        # Initialize with text predictions
        final_prob = np.array(txt_prob).copy()
        
        # Blend where images are available
        for i, idx in enumerate(img_indices):
            if i < len(img_prob) and idx < len(final_prob):
                final_prob[idx] = alpha * img_prob[i] + (1 - alpha) * txt_prob[idx]
    
    # Apply rule verification
    final_prob = verify_with_rules(task, gold_df.clean, final_prob)
    final_pred = (final_prob >= 0.5).astype(int)
    
    # Create ensemble model
    models = {'text': best_text.get('model_object')}
    if best_img:
        models['image'] = best_img.get('model_object')
    
    ensemble_model = EnsembleWrapper(
        ensemble_type='best_two',
        models=models,
        alpha=alpha,
        task=task
    )
    
    # Calculate metrics
    metrics = pack(y_true, final_prob)
    
    return {
        **metrics,
        "task": task,
        "model": f"BestTwo_alpha{alpha}",
        "model_base_name": "BestTwo",
        "model_object": ensemble_model,
        "ensemble_model": ensemble_model,
        "domain": "ensemble",
        "ensemble_type": "cross_domain",
        "prob": final_prob,
        "pred": final_pred,
        "text_model": best_text["model"],
        "image_model": best_img["model"] if best_img else None,
        "alpha": alpha
    }


def linear_blending(
    models: Dict[str, BaseEstimator],
    X_blend: np.ndarray,
    y_blend: np.ndarray,
    task: str,
    regularization: float = 1.0,
    positive: bool = True
) -> EnsembleWrapper:
    """
    Learn optimal blending weights using linear regression.
    
    Args:
        models: Dictionary of models
        X_blend: Blending set features
        y_blend: Blending set labels
        task: Task name
        regularization: Regularization strength
        positive: Constrain weights to be non-negative
        
    Returns:
        Blended ensemble
    """
    log.info(f"📊 LINEAR BLENDING for {task}")
    log.info(f"   Models: {list(models.keys())}")
    
    # Get predictions on blend set
    blend_predictions = []
    model_names = []
    
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_blend)[:, 1]
            else:
                pred = model.predict(X_blend).astype(float)
            
            blend_predictions.append(pred)
            model_names.append(name)
            
        except Exception as e:
            log.warning(f"   {name} failed on blend set: {e}")
    
    if len(blend_predictions) < 2:
        log.warning("   Not enough models for blending")
        return create_simple_average_ensemble(models, task)
    
    # Stack predictions
    blend_features = np.column_stack(blend_predictions)
    
    # Train blending model
    if positive:
        # Use non-negative least squares
        from sklearn.linear_model import Ridge
        blender = Ridge(alpha=regularization, positive=True)
    else:
        blender = Ridge(alpha=regularization)
    
    blender.fit(blend_features, y_blend)
    
    # Extract weights
    weights = blender.coef_
    intercept = blender.intercept_
    
    log.info(f"   Learned weights: {dict(zip(model_names, weights))}")
    log.info(f"   Intercept: {intercept:.3f}")
    
    # Create ensemble
    class LinearBlendingEnsemble(BaseEnsemble):
        def __init__(self, models, task, weights, intercept):
            super().__init__(models, task)
            self.blend_weights = weights
            self.intercept = intercept
        
        def predict_proba(self, X):
            predictions = []
            
            for name in model_names:  # Use same order as training
                model = self.models[name]
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X).astype(float)
                predictions.append(pred)
            
            # Apply learned weights
            blend_features = np.column_stack(predictions)
            blended_prob = np.dot(blend_features, self.blend_weights) + self.intercept
            
            # Clip to valid probability range
            blended_prob = np.clip(blended_prob, 0, 1)
            
            return np.c_[1 - blended_prob, blended_prob]
    
    ensemble = LinearBlendingEnsemble(
        {name: models[name] for name in model_names},
        task,
        weights,
        intercept
    )
    
    return EnsembleWrapper(
        ensemble_type='blending',
        models=ensemble,
        task=task,
        blending_type='linear',
        weights=weights,
        intercept=intercept
    )


def ranked_average_blending(
    models: Dict[str, BaseEstimator],
    model_scores: Dict[str, float],
    task: str,
    power: float = 2.0
) -> EnsembleWrapper:
    """
    Blend using ranked weighted average based on model scores.
    
    Args:
        models: Dictionary of models
        model_scores: Dictionary of model_name -> score (e.g., F1)
        task: Task name
        power: Power to raise scores to (higher = more weight to best models)
        
    Returns:
        Ranked average ensemble
    """
    # Calculate weights from scores
    weights = {}
    for name in models.keys():
        score = model_scores.get(name, 0.0)
        weight = score ** power
        weights[name] = weight
    
    # Normalize
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v/total_weight for k, v in weights.items()}
    else:
        # Equal weights as fallback
        weights = {k: 1.0/len(models) for k in models.keys()}
    
    log.info(f"Ranked average weights (power={power}): {weights}")
    
    # Create ensemble
    from .voting import create_voting_ensemble
    return create_voting_ensemble(
        models=models,
        task=task,
        voting='soft',
        weights=list(weights.values())
    )


def stacking_blend(
    base_models: Dict[str, BaseEstimator],
    X_blend: np.ndarray,
    y_blend: np.ndarray,
    task: str,
    meta_model: Optional[BaseEstimator] = None,
    cv_folds: int = 5,
    use_probas: bool = True
) -> EnsembleWrapper:
    """
    Create a stacking ensemble with a meta-learner.
    
    Args:
        base_models: Dictionary of base models
        X_blend: Blending set features
        y_blend: Blending set labels
        task: Task name
        meta_model: Meta-learner (default: LogisticRegression)
        cv_folds: Number of CV folds for meta-features
        use_probas: Use probabilities vs predictions
        
    Returns:
        Stacking ensemble
    """
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42)
    
    log.info(f"🏗️  STACKING BLEND for {task}")
    log.info(f"   Base models: {list(base_models.keys())}")
    log.info(f"   Meta-learner: {type(meta_model).__name__}")
    
    # Generate meta-features
    meta_features = []
    
    for name, model in base_models.items():
        try:
            if use_probas and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_blend)[:, 1]
            else:
                pred = model.predict(X_blend).astype(float)
            
            meta_features.append(pred)
            
        except Exception as e:
            log.warning(f"   {name} failed: {e}")
    
    if len(meta_features) < 2:
        log.warning("   Not enough base models for stacking")
        return create_simple_average_ensemble(base_models, task)
    
    # Stack features
    X_meta = np.column_stack(meta_features)
    
    # Train meta-learner with cross-validation
    cv_scores = cross_val_score(meta_model, X_meta, y_blend, cv=cv_folds, scoring='f1')
    log.info(f"   Meta-learner CV F1: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Train final meta-learner
    meta_model.fit(X_meta, y_blend)
    
    # Create stacking ensemble
    class StackingEnsemble(BaseEnsemble):
        def __init__(self, base_models, meta_model, task, use_probas):
            super().__init__(base_models, task)
            self.meta_model = meta_model
            self.use_probas = use_probas
            self.base_model_names = list(base_models.keys())
        
        def predict_proba(self, X):
            # Get base model predictions
            meta_features = []
            
            for name in self.base_model_names:
                model = self.models[name]
                
                if self.use_probas and hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X).astype(float)
                
                meta_features.append(pred)
            
            # Stack and predict
            X_meta = np.column_stack(meta_features)
            
            if hasattr(self.meta_model, 'predict_proba'):
                return self.meta_model.predict_proba(X_meta)
            else:
                pred = self.meta_model.predict(X_meta)
                return np.c_[1 - pred, pred]
    
    ensemble = StackingEnsemble(base_models, meta_model, task, use_probas)
    
    return EnsembleWrapper(
        ensemble_type='stacking',
        models=ensemble,
        task=task,
        meta_learner=meta_model
    )


def dynamic_blending(
    models: Dict[str, BaseEstimator],
    X_blend: np.ndarray,
    y_blend: np.ndarray,
    task: str,
    confidence_threshold: float = 0.7
) -> EnsembleWrapper:
    """
    Dynamically blend models based on prediction confidence.
    
    Models with higher confidence get more weight in the final prediction.
    
    Args:
        models: Dictionary of models
        X_blend: Blending features (for analysis)
        y_blend: Blending labels (for analysis)
        task: Task name
        confidence_threshold: Threshold for high confidence
        
    Returns:
        Dynamic blending ensemble
    """
    log.info(f"🔄 DYNAMIC BLENDING for {task}")
    
    # Analyze model confidence patterns
    confidence_stats = {}
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_blend)[:, 1]
            # Calculate confidence as distance from 0.5
            confidence = np.abs(probs - 0.5) * 2
            avg_confidence = np.mean(confidence)
            high_conf_ratio = np.mean(confidence > confidence_threshold)
            
            confidence_stats[name] = {
                'avg_confidence': avg_confidence,
                'high_conf_ratio': high_conf_ratio
            }
            
            log.info(f"   {name}: avg_conf={avg_confidence:.3f}, high_conf={high_conf_ratio:.2%}")
    
    class DynamicBlendingEnsemble(BaseEnsemble):
        def __init__(self, models, task, confidence_threshold):
            super().__init__(models, task)
            self.confidence_threshold = confidence_threshold
        
        def predict_proba(self, X):
            predictions = []
            confidences = []
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[:, 1]
                    # Calculate confidence
                    conf = np.abs(prob - 0.5) * 2
                else:
                    prob = model.predict(X).astype(float)
                    conf = np.ones_like(prob)  # Assume high confidence for binary predictions
                
                predictions.append(prob)
                confidences.append(conf)
            
            predictions = np.array(predictions)  # Shape: (n_models, n_samples)
            confidences = np.array(confidences)
            
            # Dynamic weighting based on confidence
            final_probs = []
            
            for i in range(X.shape[0]):
                sample_preds = predictions[:, i]
                sample_confs = confidences[:, i]
                
                # Weight by confidence
                weights = sample_confs / (np.sum(sample_confs) + 1e-10)
                weighted_prob = np.dot(weights, sample_preds)
                
                final_probs.append(weighted_prob)
            
            final_probs = np.array(final_probs)
            return np.c_[1 - final_probs, final_probs]
    
    ensemble = DynamicBlendingEnsemble(models, task, confidence_threshold)
    
    return EnsembleWrapper(
        ensemble_type='blending',
        models=ensemble,
        task=task,
        blending_type='dynamic',
        confidence_threshold=confidence_threshold
    )


def trimmed_mean_blending(
    models: Dict[str, BaseEstimator],
    task: str,
    trim_proportion: float = 0.1
) -> EnsembleWrapper:
    """
    Blend using trimmed mean to reduce outlier influence.
    
    Args:
        models: Dictionary of models
        task: Task name
        trim_proportion: Proportion to trim from each end
        
    Returns:
        Trimmed mean ensemble
    """
    from scipy import stats
    
    class TrimmedMeanEnsemble(BaseEnsemble):
        def __init__(self, models, task, trim_proportion):
            super().__init__(models, task)
            self.trim_proportion = trim_proportion
        
        def predict_proba(self, X):
            predictions = []
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[:, 1]
                else:
                    prob = model.predict(X).astype(float)
                
                predictions.append(prob)
            
            predictions = np.array(predictions)  # Shape: (n_models, n_samples)
            
            # Calculate trimmed mean for each sample
            final_probs = []
            for i in range(X.shape[0]):
                sample_preds = predictions[:, i]
                trimmed_mean = stats.trim_mean(sample_preds, self.trim_proportion)
                final_probs.append(trimmed_mean)
            
            final_probs = np.array(final_probs)
            return np.c_[1 - final_probs, final_probs]
    
    ensemble = TrimmedMeanEnsemble(models, task, trim_proportion)
    
    return EnsembleWrapper(
        ensemble_type='blending',
        models=ensemble,
        task=task,
        blending_type='trimmed_mean',
        trim_proportion=trim_proportion
    )


def create_simple_average_ensemble(
    models: Dict[str, BaseEstimator],
    task: str
) -> EnsembleWrapper:
    """
    Create a simple averaging ensemble as fallback.
    
    Args:
        models: Dictionary of models
        task: Task name
        
    Returns:
        Simple averaging ensemble
    """
    from .voting import create_voting_ensemble
    
    return create_voting_ensemble(
        models=models,
        task=task,
        voting='soft',
        weights=None  # Equal weights
    )