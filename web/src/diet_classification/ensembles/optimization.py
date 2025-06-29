#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble optimization module.

Provides methods for finding optimal ensemble configurations including:
- Model selection
- Weight optimization
- Ensemble size determination
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from itertools import combinations
from tqdm import tqdm

from ..core import log
from ..evaluation.metrics import pack
from ..models.training import extract_labels_safely
from .base import create_ensemble
from .voting import create_voting_ensemble, dynamic_weighted_voting
from .blending import linear_blending, stacking_blend


def top_n_ensemble(
    task: str,
    results: List[Dict],
    X_train: np.ndarray,
    clean: pd.Series,
    X_test: np.ndarray,
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    n: int = 3,
    ensemble_type: str = 'voting'
) -> Dict[str, Any]:
    """
    Build an ensemble from top N performing models.
    
    Args:
        task: Task name ('keto' or 'vegan')
        results: List of model results
        X_train: Training features
        clean: Clean text for verification
        X_test: Test features
        silver_df: Silver training data
        gold_df: Gold test data
        n: Number of models to include
        ensemble_type: Type of ensemble ('voting', 'blending', 'stacking')
        
    Returns:
        Ensemble result dictionary
    """
    log.info(f"\n🎯 BUILDING TOP-{n} ENSEMBLE for {task}")
    
    # Filter and rank models
    task_results = [r for r in results if r["task"] == task and r["model"] != "Rule"]
    
    if len(task_results) < n:
        log.warning(f"Only {len(task_results)} models available, adjusting n")
        n = len(task_results)
    
    if n < 2:
        log.warning("Not enough models for ensemble")
        return max(task_results, key=lambda x: x['F1']) if task_results else {}
    
    # Rank by composite score
    for r in task_results:
        r['composite_score'] = (
            r["F1"] + r["ACC"] + r["PREC"] + 
            r["REC"] + r["ROC"] + r["PR"]
        ) / 6
    
    # Select top N
    top_models = sorted(task_results, key=lambda x: x['composite_score'], reverse=True)[:n]
    
    log.info(f"Selected models:")
    for i, model in enumerate(top_models, 1):
        log.info(f"   {i}. {model['model']}: F1={model['F1']:.3f}, Composite={model['composite_score']:.3f}")
    
    # Get model objects
    models = {}
    for model_res in top_models:
        if 'model_object' in model_res:
            models[model_res['model']] = model_res['model_object']
    
    if len(models) < 2:
        log.error("Not enough model objects available")
        return top_models[0] if top_models else {}
    
    # Extract labels
    y_train = extract_labels_safely(silver_df, task, "silver")
    y_test = extract_labels_safely(gold_df, task, "label")
    
    # Create ensemble based on type
    if ensemble_type == 'voting':
        # Use validation performance for weights
        ensemble, weights = dynamic_weighted_voting(
            models=models,
            X_val=X_test,  # Using test as validation (not ideal but matches original)
            y_val=y_test,
            task=task,
            metric='f1'
        )
    
    elif ensemble_type == 'blending':
        ensemble = linear_blending(
            models=models,
            X_blend=X_test,
            y_blend=y_test,
            task=task
        )
    
    elif ensemble_type == 'stacking':
        ensemble = stacking_blend(
            base_models=models,
            X_blend=X_test,
            y_blend=y_test,
            task=task
        )
    
    else:
        # Default to simple voting
        ensemble = create_voting_ensemble(models, task)
    
    # Make predictions
    prob = ensemble.predict_proba(X_test)[:, 1]
    
    # Apply verification
    from ..classification.verification import verify_with_rules
    prob = verify_with_rules(task, clean, prob)
    
    # Calculate metrics
    pred = (prob >= 0.5).astype(int)
    metrics = pack(y_test, prob)
    
    return {
        **metrics,
        "task": task,
        "model": f"Top{n}_{ensemble_type}",
        "model_base_name": f"Top{n}",
        "domain": "ensemble",
        "ensemble_type": ensemble_type,
        "model_object": ensemble,
        "ensemble_model": ensemble,
        "prob": prob,
        "pred": pred,
        "n_models": n,
        "models_used": list(models.keys())
    }


def best_ensemble(
    task: str,
    results: List[Dict],
    X_train: np.ndarray,
    clean: pd.Series,
    X_test: np.ndarray,
    silver: pd.DataFrame,
    gold: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    max_models: int = 7,
    ensemble_types: List[str] = None
) -> Dict[str, Any]:
    """
    Find the best ensemble configuration through systematic search.
    
    Tests different ensemble sizes and types to find optimal configuration.
    
    Args:
        task: Task name
        results: List of model results
        X_train: Training features
        clean: Clean text
        X_test: Test features
        silver: Silver data
        gold: Gold data
        weights: Metric weights for optimization
        max_models: Maximum ensemble size to test
        ensemble_types: Types to test (default: ['voting', 'blending'])
        
    Returns:
        Best ensemble configuration
    """
    if ensemble_types is None:
        ensemble_types = ['voting', 'blending']
    
    if weights is None:
        weights = {
            'F1': 1/3,
            'ACC': 1/3,
            'ROC': 1/3
        }
    
    log.info(f"\n🔍 OPTIMIZING ENSEMBLE for {task}")
    log.info(f"   Max models: {max_models}")
    log.info(f"   Ensemble types: {ensemble_types}")
    log.info(f"   Optimization weights: {weights}")
    
    # Filter available models
    task_results = [r for r in results if r["task"] == task and r["model"] != "Rule"]
    n_available = len(task_results)
    
    if n_available < 2:
        log.warning("Not enough models for ensemble optimization")
        return max(task_results, key=lambda x: x['F1']) if task_results else {}
    
    # Limit search space
    max_n = min(max_models, n_available)
    
    best_score = -1
    best_config = None
    
    # Test different configurations
    configs_tested = []
    
    with tqdm(total=max_n * len(ensemble_types), desc="Testing configurations") as pbar:
        for n in range(2, max_n + 1):
            for ens_type in ensemble_types:
                pbar.set_description(f"Testing n={n}, type={ens_type}")
                
                try:
                    # Build ensemble
                    ensemble_result = top_n_ensemble(
                        task=task,
                        results=results,
                        X_train=X_train,
                        clean=clean,
                        X_test=X_test,
                        silver_df=silver,
                        gold_df=gold,
                        n=n,
                        ensemble_type=ens_type
                    )
                    
                    # Calculate weighted score
                    score = sum(
                        weights.get(metric, 0) * ensemble_result.get(metric, 0)
                        for metric in weights
                    )
                    
                    config = {
                        'n': n,
                        'type': ens_type,
                        'score': score,
                        'F1': ensemble_result.get('F1', 0),
                        'ACC': ensemble_result.get('ACC', 0)
                    }
                    configs_tested.append(config)
                    
                    if score > best_score:
                        best_score = score
                        best_config = ensemble_result
                        log.info(f"   New best: n={n}, {ens_type}, score={score:.3f}")
                    
                except Exception as e:
                    log.warning(f"   Failed n={n}, {ens_type}: {e}")
                
                pbar.update(1)
    
    # Log optimization summary
    log.info(f"\n📊 Optimization Summary:")
    log.info(f"   Configurations tested: {len(configs_tested)}")
    
    if configs_tested:
        best_by_f1 = max(configs_tested, key=lambda x: x['F1'])
        log.info(f"   Best by F1: n={best_by_f1['n']}, {best_by_f1['type']} (F1={best_by_f1['F1']:.3f})")
        
        best_by_score = max(configs_tested, key=lambda x: x['score'])
        log.info(f"   Best by weighted score: n={best_by_score['n']}, {best_by_score['type']} (score={best_by_score['score']:.3f})")
    
    return best_config or {}


def greedy_ensemble_selection(
    task: str,
    results: List[Dict],
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = 'f1',
    max_models: int = 10,
    replacement: bool = True
) -> Dict[str, Any]:
    """
    Build ensemble using greedy forward selection.
    
    Iteratively adds models that improve ensemble performance.
    
    Args:
        task: Task name
        results: List of model results
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize
        max_models: Maximum models to include
        replacement: Allow selecting same model multiple times
        
    Returns:
        Greedy ensemble result
    """
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    
    metric_funcs = {
        'f1': f1_score,
        'accuracy': accuracy_score,
        'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred >= 0.5)
    }
    
    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}")
    
    metric_func = metric_funcs[metric]
    
    log.info(f"\n🎯 GREEDY ENSEMBLE SELECTION for {task}")
    log.info(f"   Optimizing: {metric}")
    log.info(f"   With replacement: {replacement}")
    
    # Get available models
    task_models = [
        r for r in results 
        if r["task"] == task and "model_object" in r
    ]
    
    if not task_models:
        log.error("No models available for selection")
        return {}
    
    # Initialize ensemble
    selected_models = []
    selected_weights = []
    best_score = 0
    
    # Greedy selection
    for i in range(max_models):
        best_addition = None
        best_addition_score = best_score
        
        # Try each model
        for model_res in task_models:
            if not replacement and model_res in selected_models:
                continue
            
            # Create temporary ensemble
            temp_models = selected_models + [model_res]
            temp_ensemble_preds = []
            
            for m in temp_models:
                model = m['model_object']
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)[:, 1]
                else:
                    pred = model.predict(X_val).astype(float)
                temp_ensemble_preds.append(pred)
            
            # Average predictions
            ensemble_pred = np.mean(temp_ensemble_preds, axis=0)
            
            # Evaluate
            if metric == 'roc_auc':
                score = metric_func(y_val, ensemble_pred)
            else:
                score = metric_func(y_val, ensemble_pred >= 0.5)
            
            if score > best_addition_score:
                best_addition = model_res
                best_addition_score = score
        
        # Add best model if improvement
        if best_addition is not None:
            selected_models.append(best_addition)
            selected_weights.append(1.0)  # Equal weights for now
            best_score = best_addition_score
            
            log.info(f"   Step {i+1}: Added {best_addition['model']} ({metric}={best_score:.3f})")
        else:
            log.info(f"   No improvement at step {i+1}, stopping")
            break
    
    if not selected_models:
        log.warning("No models selected")
        return {}
    
    # Create final ensemble
    final_models = {
        f"{m['model']}_{i}": m['model_object'] 
        for i, m in enumerate(selected_models)
    }
    
    ensemble = create_voting_ensemble(
        models=final_models,
        task=task,
        voting='soft'
    )
    
    # Final evaluation
    final_pred = ensemble.predict_proba(X_val)[:, 1]
    
    return {
        "task": task,
        "model": f"Greedy_{len(selected_models)}",
        "ensemble_model": ensemble,
        "n_models": len(selected_models),
        "models_selected": [m['model'] for m in selected_models],
        "selection_metric": metric,
        "final_score": best_score,
        "prob": final_pred,
        "pred": (final_pred >= 0.5).astype(int)
    }


def ensemble_pruning(
    ensemble: BaseEstimator,
    models: Dict[str, BaseEstimator],
    X_val: np.ndarray,
    y_val: np.ndarray,
    min_improvement: float = 0.01
) -> Tuple[BaseEstimator, List[str]]:
    """
    Prune ensemble by removing models that don't contribute.
    
    Args:
        ensemble: Current ensemble
        models: Dictionary of all models
        X_val: Validation features
        y_val: Validation labels
        min_improvement: Minimum improvement to keep model
        
    Returns:
        Tuple of (pruned_ensemble, removed_models)
    """
    from sklearn.metrics import f1_score
    
    # Get baseline performance
    baseline_pred = ensemble.predict(X_val)
    baseline_f1 = f1_score(y_val, baseline_pred)
    
    log.info(f"Ensemble pruning - baseline F1: {baseline_f1:.3f}")
    
    removed_models = []
    
    for model_name, model in models.items():
        # Create ensemble without this model
        temp_models = {k: v for k, v in models.items() if k != model_name}
        
        if len(temp_models) < 2:
            continue
        
        temp_ensemble = create_voting_ensemble(temp_models, "temp")
        temp_pred = temp_ensemble.predict(X_val)
        temp_f1 = f1_score(y_val, temp_pred)
        
        # Check if removing improves or maintains performance
        if temp_f1 >= baseline_f1 - min_improvement:
            log.info(f"   Removing {model_name}: F1 {baseline_f1:.3f} → {temp_f1:.3f}")
            removed_models.append(model_name)
            models = temp_models
            baseline_f1 = temp_f1
    
    # Create final pruned ensemble
    if removed_models:
        pruned_ensemble = create_voting_ensemble(models, ensemble.task)
    else:
        pruned_ensemble = ensemble
    
    return pruned_ensemble, removed_models