#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter tuning module for model optimization.

Provides grid search, random search, and Bayesian optimization methods.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score,
    ParameterGrid
)
from sklearn.metrics import f1_score

from ..core import log
from ..config import CFG
from ..utils.memory import optimize_memory_usage


# Hyperparameter grids for each model type
HYPER = {
    # Text models
    "Softmax": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [1000]
    },
    "SGD": {
        "alpha": [1e-4, 1e-3, 1e-2],
        "loss": ["log_loss", "modified_huber"],
        "penalty": ["l2", "elasticnet"]
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0]
    },
    "PA": {
        "C": [0.5, 1.0, 2.0],
        "loss": ["hinge", "squared_hinge"]
    },
    "NB": {
        "alpha": [0.1, 0.5, 1.0]
    },
    "LinearSVC": {
        "C": [0.1, 1.0, 10.0],
        "loss": ["hinge", "squared_hinge"]
    },
    
    # Image/mixed models
    "MLP": {
        "hidden_layer_sizes": [(100,), (256,), (512, 128)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01],
        "early_stopping": [True]
    },
    "RF": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "LGBM": {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "n_estimators": [100, 200],
        "min_child_samples": [10, 20],
        "subsample": [0.8, 1.0]
    },
    
    # SVM with RBF kernel
    "SVM_RBF": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto", 0.001, 0.01]
    }
}

# Global cache for best hyperparameters
BEST_PARAMS_CACHE = {}


def tune(
    name: str,
    base_model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 3,
    scoring: str = 'f1',
    n_jobs: int = -1,
    verbose: int = 0
) -> BaseEstimator:
    """
    Tune hyperparameters using grid search.
    
    Args:
        name: Model name (key in HYPER dict)
        base_model: Base estimator to tune
        X: Training features
        y: Training labels
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Fitted model with best hyperparameters
    """
    # Check cache first
    cache_key = f"{name}_{X.shape}_{scoring}"
    if cache_key in BEST_PARAMS_CACHE:
        log.info(f"Using cached hyperparameters for {name}")
        base_model.set_params(**BEST_PARAMS_CACHE[cache_key])
        return base_model.fit(X, y)
    
    # Get parameter grid
    param_grid = HYPER.get(name, {})
    
    if not param_grid:
        log.info(f"No hyperparameters defined for {name}, using defaults")
        return base_model.fit(X, y)
    
    # Calculate total fits
    n_combinations = len(list(ParameterGrid(param_grid)))
    total_fits = n_combinations * cv
    
    log.info(f"🔍 Tuning {name}")
    log.info(f"   ├─ Parameter combinations: {n_combinations}")
    log.info(f"   ├─ CV folds: {cv}")
    log.info(f"   └─ Total fits: {total_fits}")
    
    # Perform grid search
    try:
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        # Fit with timing
        start_time = time.time()
        grid_search.fit(X, y)
        search_time = time.time() - start_time
        
        # Get results
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        log.info(f"   ✅ Best score: {best_score:.3f}")
        log.info(f"   ✅ Best params: {best_params}")
        log.info(f"   ✅ Search time: {search_time:.1f}s")
        
        # Cache best parameters
        BEST_PARAMS_CACHE[cache_key] = best_params
        
        # Save to disk
        save_best_params(name, best_params)
        
        return grid_search.best_estimator_
        
    except Exception as e:
        log.error(f"Grid search failed for {name}: {e}")
        log.info("Falling back to default parameters")
        return base_model.fit(X, y)


def tune_with_early_stopping(
    name: str,
    base_model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 3,
    patience: int = 3,
    min_improvement: float = 0.001
) -> BaseEstimator:
    """
    Tune with early stopping to save computation.
    
    Stops when no improvement for 'patience' iterations.
    
    Args:
        name: Model name
        base_model: Base estimator
        X: Training features
        y: Training labels
        cv: CV folds
        patience: Iterations without improvement before stopping
        min_improvement: Minimum improvement required
        
    Returns:
        Tuned model
    """
    param_grid = HYPER.get(name, {})
    
    if not param_grid:
        return base_model.fit(X, y)
    
    # Convert to list for iteration
    param_list = list(ParameterGrid(param_grid))
    
    log.info(f"🔍 Tuning {name} with early stopping")
    log.info(f"   ├─ Total combinations: {len(param_list)}")
    log.info(f"   └─ Patience: {patience}")
    
    best_score = -np.inf
    best_params = None
    best_model = None
    patience_counter = 0
    
    with tqdm(param_list, desc=f"Tuning {name}") as pbar:
        for i, params in enumerate(pbar):
            try:
                # Clone and set parameters
                model = clone(base_model).set_params(**params)
                
                # Cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
                mean_score = scores.mean()
                
                pbar.set_postfix({
                    'best': f"{best_score:.3f}",
                    'current': f"{mean_score:.3f}",
                    'patience': patience_counter
                })
                
                # Check improvement
                if mean_score > best_score + min_improvement:
                    best_score = mean_score
                    best_params = params
                    best_model = model.fit(X, y)
                    patience_counter = 0
                    
                    log.info(f"   New best: {mean_score:.3f} with {params}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    log.info(f"   Early stopping after {i+1} iterations")
                    break
                    
            except Exception as e:
                log.debug(f"   Params {params} failed: {e}")
                patience_counter += 1
    
    if best_model is not None:
        # Cache and save
        cache_key = f"{name}_{X.shape}_f1"
        BEST_PARAMS_CACHE[cache_key] = best_params
        save_best_params(name, best_params)
        
        return best_model
    else:
        log.warning(f"No valid configuration found for {name}")
        return base_model.fit(X, y)


def random_search_tune(
    name: str,
    base_model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42
) -> BaseEstimator:
    """
    Tune hyperparameters using random search.
    
    More efficient than grid search for large parameter spaces.
    
    Args:
        name: Model name
        base_model: Base estimator
        X: Training features
        y: Training labels
        n_iter: Number of iterations
        cv: CV folds
        random_state: Random seed
        
    Returns:
        Tuned model
    """
    param_distributions = HYPER.get(name, {})
    
    if not param_distributions:
        return base_model.fit(X, y)
    
    log.info(f"🎲 Random search for {name}")
    log.info(f"   ├─ Iterations: {n_iter}")
    log.info(f"   └─ CV folds: {cv}")
    
    try:
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
        
        random_search.fit(X, y)
        
        log.info(f"   ✅ Best score: {random_search.best_score_:.3f}")
        log.info(f"   ✅ Best params: {random_search.best_params_}")
        
        # Cache and save
        cache_key = f"{name}_{X.shape}_f1"
        BEST_PARAMS_CACHE[cache_key] = random_search.best_params_
        save_best_params(name, random_search.best_params_)
        
        return random_search.best_estimator_
        
    except Exception as e:
        log.error(f"Random search failed: {e}")
        return base_model.fit(X, y)


def adaptive_tuning(
    name: str,
    base_model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    time_budget: float = 60.0
) -> BaseEstimator:
    """
    Adaptively tune based on time budget.
    
    Automatically chooses between grid search, random search,
    or default based on available time.
    
    Args:
        name: Model name
        base_model: Base estimator
        X: Training features
        y: Training labels
        time_budget: Time budget in seconds
        
    Returns:
        Tuned model
    """
    param_grid = HYPER.get(name, {})
    
    if not param_grid:
        return base_model.fit(X, y)
    
    # Estimate time per fit
    start = time.time()
    _ = clone(base_model).fit(X[:min(1000, len(X))], y[:min(1000, len(y))])
    time_per_fit = time.time() - start
    
    # Calculate possible fits
    n_combinations = len(list(ParameterGrid(param_grid)))
    cv_folds = 3
    total_grid_time = n_combinations * cv_folds * time_per_fit
    
    log.info(f"⏰ Adaptive tuning for {name}")
    log.info(f"   ├─ Time budget: {time_budget:.1f}s")
    log.info(f"   ├─ Estimated grid search time: {total_grid_time:.1f}s")
    
    if total_grid_time <= time_budget:
        # Full grid search
        log.info(f"   └─ Using grid search")
        return tune(name, base_model, X, y)
    
    elif total_grid_time <= time_budget * 10:
        # Random search
        n_iter = int(time_budget / (cv_folds * time_per_fit))
        log.info(f"   └─ Using random search with {n_iter} iterations")
        return random_search_tune(name, base_model, X, y, n_iter=n_iter)
    
    else:
        # Default parameters
        log.info(f"   └─ Using default parameters (too many combinations)")
        return base_model.fit(X, y)


def save_best_params(name: str, params: Dict[str, Any]):
    """Save best parameters to disk."""
    params_file = CFG.artifacts_dir / "best_hyperparams.json"
    
    try:
        if params_file.exists():
            with open(params_file, 'r') as f:
                all_params = json.load(f)
        else:
            all_params = {}
        
        all_params[name] = params
        
        with open(params_file, 'w') as f:
            json.dump(all_params, f, indent=2)
            
    except Exception as e:
        log.warning(f"Could not save hyperparameters: {e}")


def load_best_params(name: str) -> Optional[Dict[str, Any]]:
    """Load best parameters from disk."""
    params_file = CFG.artifacts_dir / "best_hyperparams.json"
    
    if params_file.exists():
        try:
            with open(params_file, 'r') as f:
                all_params = json.load(f)
            return all_params.get(name)
        except Exception as e:
            log.warning(f"Could not load hyperparameters: {e}")
    
    return None


def get_param_importance(
    name: str,
    base_model: BaseEstimator,
    param_grid: Dict[str, list],
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 3
) -> Dict[str, float]:
    """
    Analyze importance of each hyperparameter.
    
    Args:
        name: Model name
        base_model: Base estimator
        param_grid: Parameter grid
        X: Features
        y: Labels
        n_repeats: Number of repeats for stability
        
    Returns:
        Dictionary of parameter -> importance score
    """
    log.info(f"📊 Analyzing parameter importance for {name}")
    
    importance_scores = {}
    
    for param_name, param_values in param_grid.items():
        if len(param_values) < 2:
            continue
        
        scores = []
        
        for _ in range(n_repeats):
            # Score with each parameter value
            param_scores = []
            
            for value in param_values:
                try:
                    model = clone(base_model)
                    model.set_params(**{param_name: value})
                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1')
                    param_scores.append(cv_scores.mean())
                except Exception:
                    param_scores.append(0)
            
            # Calculate variance in scores
            if param_scores:
                score_variance = np.var(param_scores)
                scores.append(score_variance)
        
        # Average importance across repeats
        importance_scores[param_name] = np.mean(scores) if scores else 0
    
    # Normalize scores
    total = sum(importance_scores.values())
    if total > 0:
        importance_scores = {k: v/total for k, v in importance_scores.items()}
    
    # Log results
    for param, importance in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
        log.info(f"   ├─ {param}: {importance:.3f}")
    
    return importance_scores