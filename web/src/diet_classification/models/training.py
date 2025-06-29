#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training module with complete Mode A implementation.

This module implements the full training pipeline including:
- SMOTE for class balancing
- Checkpoint/resume capability
- Progress tracking
- Comprehensive evaluation
"""

import logging
import os

import time
import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm


from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score

from ..core import log, get_pipeline_state
from ..config import CFG
from ..models.builders import build_models
from ..models.tuning import tune, tune_with_early_stopping
from ..features.combiners import apply_smote, combine_features
from ..evaluation.metrics import pack, table, ensure_predict_proba
from ..classification.verification import verify_with_rules
from ..utils.memory import optimize_memory_usage
from ..utils.validation import tune_threshold
from ..evaluation.export import (
    export_results_to_csv, 
    log_false_predictions,
    save_model_artifacts
)


def run_mode_A(
    X_silver,
    gold_clean: pd.Series,
    X_gold,
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    domain: str = "text",
    apply_smote_flag: bool = True,
    checkpoint_dir: Path = None,
    use_early_stopping: bool = False,
    save_artifacts: bool = True
) -> List[Dict[str, Any]]:
    """
    Train models on silver (weak) labels and evaluate on gold standard.
    
    This is the complete implementation matching the original functionality:
    - Handles checkpoint/resume
    - Applies SMOTE for class balancing
    - Trains multiple model types
    - Comprehensive evaluation with all metrics
    - Progress tracking and memory optimization
    
    Args:
        X_silver: Training features (sparse or dense matrix)
        gold_clean: Normalized gold text for rule verification
        X_gold: Test features
        silver_df: Silver training DataFrame
        gold_df: Gold test DataFrame
        domain: Feature domain ('text', 'image', or 'both')
        apply_smote_flag: Whether to apply SMOTE for class balancing
        checkpoint_dir: Directory for checkpoints (defaults to CFG.checkpoints_dir)
        use_early_stopping: Use early stopping for hyperparameter tuning
        save_artifacts: Whether to save model artifacts
        
    Returns:
        List of result dictionaries with metrics and predictions
    """
    # Setup
    pipeline_state = get_pipeline_state()
    if checkpoint_dir is None:
        checkpoint_dir = CFG.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Suppress convergence warnings
    warnings.filterwarnings('ignore', message='.*Liblinear failed to converge.*')
    warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')
    
    # Initialize tracking
    results: List[Dict[str, Any]] = []
    pipeline_start = time.time()
    
    # Log initialization
    log.info(f"\n🚀 STARTING MODE A TRAINING PIPELINE")
    log.info(f"   ├─ Domain: {domain}")
    log.info(f"   ├─ SMOTE enabled: {apply_smote_flag}")
    log.info(f"   ├─ Early stopping: {use_early_stopping}")
    log.info(f"   ├─ Silver samples: {len(silver_df):,}")
    log.info(f"   ├─ Gold samples: {len(gold_df):,}")
    log.info(f"   ├─ Feature shape: {X_silver.shape}")
    log.info(f"   └─ Checkpoint dir: {checkpoint_dir}")
    
    # Memory check
    memory_status = optimize_memory_usage("Training start")
    if memory_status == "critical":
        log.warning("   ⚠️  Critical memory - enabling minimal mode")
        os.environ['FORCE_MINIMAL_MODELS'] = '1'
    
    # Check for existing checkpoints
    checkpoint_state = load_training_checkpoint(checkpoint_dir, domain)
    
    if checkpoint_state['completed_tasks']:
        log.info(f"   📂 Resuming from checkpoint - completed: {checkpoint_state['completed_tasks']}")
        results.extend(checkpoint_state['loaded_results'])
    
    # Show class distribution
    log.info(f"\n📊 Class Distribution:")
    for task in ["keto", "vegan"]:
        if f"silver_{task}" in silver_df.columns:
            silver_pos = silver_df[f"silver_{task}"].sum()
            silver_rate = silver_pos / len(silver_df) * 100
        else:
            silver_pos = silver_rate = 0
            
        if f"label_{task}" in gold_df.columns:
            gold_pos = gold_df[f"label_{task}"].sum()
            gold_rate = gold_pos / len(gold_df) * 100
        else:
            gold_pos = gold_rate = 0
            
        log.info(f"   {task:>5} - Silver: {silver_pos:,}/{len(silver_df):,} ({silver_rate:.1f}%) | "
                 f"Gold: {gold_pos:,}/{len(gold_df):,} ({gold_rate:.1f}%)")
    
    # Main training loop
    task_progress = tqdm(["keto", "vegan"], desc="🔬 Training Tasks",
                         position=0, leave=True,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for task in task_progress:
        # Skip if already completed
        if task in checkpoint_state['completed_tasks']:
            log.info(f"\n⏭️  Skipping {task} - already completed")
            task_progress.update(1)
            continue
            
        task_start = time.time()
        task_progress.set_description(f"🔬 Training {task.capitalize()}")
        
        # Extract labels safely
        y_train = extract_labels_safely(silver_df, task, "silver")
        y_true = extract_labels_safely(gold_df, task, "label")
        
        log.info(f"\n🎯 Processing {task.upper()} classification:")
        log.info(f"   ├─ Training samples: {len(y_train):,} (Positive: {y_train.sum():,}, {y_train.mean():.1%})")
        log.info(f"   └─ Test samples: {len(y_true):,} (Positive: {y_true.sum():,}, {y_true.mean():.1%})")
        
        # Handle class imbalance with SMOTE
        X_train = X_silver
        if apply_smote_flag and domain != "image":  # SMOTE works better with text features
            log.info(f"   🔄 Applying SMOTE for class balancing...")
            try:
                X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
                
                if len(y_train_balanced) != len(y_train):
                    log.info(f"      ├─ Original: {len(y_train):,} samples")
                    log.info(f"      ├─ After SMOTE: {len(y_train_balanced):,} samples")
                    log.info(f"      └─ Positive rate: {y_train.mean():.1%} → {y_train_balanced.mean():.1%}")
                    X_train = X_train_balanced
                    y_train = y_train_balanced
                else:
                    log.info(f"      └─ No resampling needed (balanced enough)")
                    
            except Exception as e:
                log.warning(f"   ⚠️  SMOTE failed: {e}")
                log.info(f"   └─ Continuing with original imbalanced data")
        
        # Build models for this domain
        models = build_models(task, domain)
        
        # Filter out already completed models
        completed_for_task = checkpoint_state['completed_models'].get(task, [])
        if completed_for_task:
            log.info(f"   📂 Found {len(completed_for_task)} completed models")
            models = {k: v for k, v in models.items() 
                     if f"{k}_{domain.upper()}" not in completed_for_task}
        
        if not models:
            log.info(f"   ✅ All models already trained for {task}")
            task_progress.update(1)
            continue
            
        log.info(f"   🤖 Training {len(models)} models: {list(models.keys())}")
        
        # Model training
        task_results = []
        best_f1 = -1.0
        best_result = None
        
        model_progress = tqdm(models.items(), desc="   ├─ Models",
                              position=1, leave=False,
                              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")
        
        for name, base_model in model_progress:
            model_start = time.time()
            model_progress.set_description(f"   ├─ Training {name}")
            
            try:
                # Hyperparameter tuning
                if use_early_stopping and name in ["RF", "LGBM", "MLP"]:
                    tuned_model = tune_with_early_stopping(
                        name, base_model, X_train, y_train
                    )
                else:
                    tuned_model = tune(name, base_model, X_train, y_train)
                
                # Ensure model has predict_proba
                tuned_model = ensure_predict_proba(tuned_model, X_train, y_train)
                
                # Make predictions
                prob = tuned_model.predict_proba(X_gold)[:, 1]
                
                # Apply rule-based verification
                prob = verify_with_rules(task, gold_clean, prob)
                
                # Calculate metrics
                pred = (prob >= 0.5).astype(int)
                metrics = pack(y_true, prob)
                
                # Create result dictionary
                model_name = f"{name}_{domain.upper()}"
                result = {
                    "task": task,
                    "model": model_name,
                    "model_base_name": name,
                    "domain": domain,
                    "model_object": tuned_model,
                    "prob": prob,
                    "pred": pred,
                    "training_time": time.time() - model_start,
                    "n_samples_train": len(y_train),
                    "n_samples_test": len(y_true),
                    "feature_dimensions": X_train.shape,
                    **metrics
                }
                
                task_results.append(result)
                
                # Track best model
                if result['F1'] > best_f1:
                    best_f1 = result['F1']
                    best_result = result
                
                # Update progress
                model_progress.set_postfix({
                    'F1': f"{result['F1']:.3f}",
                    'ACC': f"{result['ACC']:.3f}",
                    'Time': f"{result['training_time']:.1f}s"
                })
                
                # Log results
                log.info(f"      ✅ {name}: F1={result['F1']:.3f}, ACC={result['ACC']:.3f}, "
                        f"Time={result['training_time']:.1f}s")
                
                # Log false predictions for debugging
                if save_artifacts:
                    log_false_predictions(
                        task=task,
                        texts=gold_clean,
                        y_true=y_true,
                        y_pred=pred,
                        model_name=model_name,
                        output_dir=CFG.artifacts_dir / "predictions"
                    )
                
            except Exception as e:
                log.error(f"      ❌ {name} failed: {str(e)[:80]}...")
                if log.level <= logging.DEBUG:
                    import traceback
                    log.debug(f"Full traceback:\n{traceback.format_exc()}")
                continue
        
        # Task summary
        task_time = time.time() - task_start
        
        if task_results:
            log.info(f"\n   📊 {task.upper()} Results Summary:")
            log.info(f"   ├─ Models trained: {len(task_results)}")
            log.info(f"   ├─ Best model: {best_result['model']} (F1={best_f1:.3f})")
            log.info(f"   └─ Task time: {task_time:.1f}s")
            
            # Add results
            results.extend(task_results)
            
            # Save checkpoint
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                task=task,
                domain=domain,
                results=task_results,
                all_results=results,
                task_time=task_time
            )
            
            # Save best model to pipeline state
            if best_result:
                model_key = f"{task}_{domain}_{best_result['model_base_name']}"
                pipeline_state.models[model_key] = best_result['model_object']
                
        else:
            log.warning(f"   ⚠️  No models successfully trained for {task}")
        
        # Memory cleanup
        optimize_memory_usage(f"{task} training complete")
        
        # Update main progress
        task_progress.update(1)
    
    # Pipeline completion
    pipeline_time = time.time() - pipeline_start
    
    log.info(f"\n🏁 MODE A PIPELINE COMPLETE:")
    log.info(f"   ├─ Total time: {pipeline_time:.1f}s")
    log.info(f"   ├─ Tasks completed: 2")
    log.info(f"   ├─ Models trained: {len(results)}")
    log.info(f"   └─ Domain: {domain}")
    
    # Export results
    if results and save_artifacts:
        # Prepare metadata
        metadata = {
            'pipeline_time': pipeline_time,
            'domain': domain,
            'silver_size': len(silver_df),
            'gold_size': len(gold_df),
            'feature_dimensions': X_silver.shape,
            'smote_applied': apply_smote_flag,
            'timestamp': datetime.now().isoformat()
        }
        
        # Export to CSV
        export_results_to_csv(results, metadata, domain)
        
        # Save model artifacts
        save_model_artifacts(results, domain)
    
    # Display results table
    if results:
        table("MODE A Results", results)
    
    return results


def extract_labels_safely(
    df: pd.DataFrame, 
    task: str, 
    prefix: str = "label"
) -> np.ndarray:
    """
    Safely extract labels from DataFrame with error handling.
    
    Args:
        df: DataFrame containing label columns
        task: Task name ('keto' or 'vegan')
        prefix: Column prefix ('label' or 'silver')
        
    Returns:
        NumPy array of integer labels
    """
    col_name = f"{prefix}_{task}"
    
    if col_name not in df.columns:
        log.error(f"Missing column: {col_name}")
        # Return appropriate default based on task
        default_val = 0 if task == "keto" else 1
        return np.full(len(df), default_val, dtype=int)
    
    labels = df[col_name].copy()
    
    # Handle NaN values
    if labels.isna().any():
        nan_count = labels.isna().sum()
        log.warning(f"Found {nan_count} NaN values in {col_name}, filling with defaults")
        default_val = 0 if task == "keto" else 1
        labels = labels.fillna(default_val)
    
    # Convert to int
    try:
        labels = labels.astype(int)
    except ValueError as e:
        log.error(f"Cannot convert {col_name} to int: {e}")
        labels = pd.to_numeric(labels, errors='coerce').fillna(
            0 if task == "keto" else 1
        ).astype(int)
    
    return labels.values


def load_training_checkpoint(
    checkpoint_dir: Path, 
    domain: str
) -> Dict[str, Any]:
    """
    Load existing training checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        domain: Feature domain
        
    Returns:
        Dictionary with checkpoint state
    """
    checkpoint_state = {
        'completed_tasks': set(),
        'completed_models': {},
        'loaded_results': []
    }
    
    # Look for existing checkpoints
    checkpoint_pattern = f"checkpoint_*_{domain}.pkl"
    existing_checkpoints = sorted(checkpoint_dir.glob(checkpoint_pattern))
    
    if not existing_checkpoints:
        return checkpoint_state
    
    log.info(f"   📂 Found {len(existing_checkpoints)} existing checkpoints")
    
    for ckpt_path in existing_checkpoints:
        try:
            with open(ckpt_path, 'rb') as f:
                ckpt_data = pickle.load(f)
            
            task = ckpt_data['task']
            checkpoint_state['completed_tasks'].add(task)
            
            # Load results
            if 'all_results' in ckpt_data:
                checkpoint_state['loaded_results'].extend(ckpt_data['all_results'])
                
                # Track completed models
                completed_models = [r['model'] for r in ckpt_data['all_results']]
                checkpoint_state['completed_models'][task] = completed_models
                
                log.info(f"   ├─ Loaded {task} checkpoint: {len(completed_models)} models")
                
        except Exception as e:
            log.warning(f"   ├─ Failed to load {ckpt_path.name}: {e}")
    
    return checkpoint_state


def save_training_checkpoint(
    checkpoint_dir: Path,
    task: str,
    domain: str,
    results: List[Dict],
    all_results: List[Dict],
    task_time: float
):
    """
    Save training checkpoint for resume capability.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        task: Task name
        domain: Feature domain
        results: Task-specific results
        all_results: All results so far
        task_time: Time taken for this task
    """
    checkpoint_path = checkpoint_dir / f"checkpoint_{task}_{domain}.pkl"
    
    checkpoint_data = {
        'task': task,
        'domain': domain,
        'task_results': results,
        'all_results': all_results,
        'timestamp': datetime.now().isoformat(),
        'task_time': task_time,
        'completed_models': [r['model'] for r in results]
    }
    
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        log.info(f"   💾 Checkpoint saved: {checkpoint_path.name}")
        
        # Also update pipeline state checkpoint
        pipeline_state = get_pipeline_state()
        pipeline_state.save_checkpoint(f"{task}_{domain}", checkpoint_data)
        
    except Exception as e:
        log.error(f"   ❌ Failed to save checkpoint: {e}")


def train_single_model(
    model_name: str,
    model: BaseEstimator,
    X_train,
    y_train,
    X_test,
    y_test,
    task: str,
    use_tuning: bool = True
) -> Dict[str, Any]:
    """
    Train a single model with evaluation.
    
    Convenience function for training individual models outside the main pipeline.
    
    Args:
        model_name: Name of the model
        model: Model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        task: Task name
        use_tuning: Whether to use hyperparameter tuning
        
    Returns:
        Result dictionary with metrics
    """
    log.info(f"Training {model_name} for {task}")
    
    start_time = time.time()
    
    # Tune or train directly
    if use_tuning:
        model = tune(model_name, model, X_train, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Ensure predict_proba
    model = ensure_predict_proba(model, X_train, y_train)
    
    # Predict
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = pack(y_test, prob)
    
    training_time = time.time() - start_time
    
    return {
        "task": task,
        "model": model_name,
        "model_object": model,
        "prob": prob,
        "pred": pred,
        "training_time": training_time,
        **metrics
    }









# def run_mode_A(
#     X_silver,
#     gold_clean: pd.Series,
#     X_gold,
#     silver_df: pd.DataFrame,
#     gold_df: pd.DataFrame,
#     *,
#     domain: str = "text",
#     apply_smote_flag: bool = True,
#     checkpoint_dir: Path = None
# ) -> list[dict]:
#     """
#     Train on weak (silver) labels, evaluate on gold standard labels.

#     Enhanced with complete checkpoint/resume functionality.
#     """
#     # Use persistent checkpoint directory
#     if checkpoint_dir is None:
#         checkpoint_dir = CFG.checkpoints_dir
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)

#     warnings.filterwarnings('ignore', message='.*truth value of an array.*')

#     # Ensure all arrays are properly formatted
#     if hasattr(X_silver, 'toarray'):
#         X_silver = X_silver.toarray() if X_silver.nnz < 1e6 else X_silver

#     # Initialize results and timing
#     results: list[dict] = []
#     pipeline_start = time.time()

#     # Check for existing checkpoints and load completed work
#     checkpoint_state = {
#         'completed_tasks': set(),
#         'completed_models': {},  # task -> list of completed model names
#         'loaded_results': []
#     }

#     existing_checkpoints = sorted(
#         checkpoint_dir.glob(f"checkpoint_*_{domain}.pkl"))

#     if existing_checkpoints:
#         log.info(
#             f"   📂 Found {len(existing_checkpoints)} existing checkpoints")

#         for ckpt_path in existing_checkpoints:
#             try:
#                 with open(ckpt_path, 'rb') as f:
#                     ckpt_data = pickle.load(f)

#                 task = ckpt_data['task']
#                 checkpoint_state['completed_tasks'].add(task)

#                 # Load completed models for this task
#                 if 'all_results' in ckpt_data:
#                     checkpoint_state['loaded_results'].extend(
#                         ckpt_data['all_results'])

#                     # Track which models were completed
#                     completed_model_names = [r['model']
#                                              for r in ckpt_data['all_results']]
#                     checkpoint_state['completed_models'][task] = completed_model_names

#                     log.info(
#                         f"   ├─ Loaded {task} checkpoint: {len(completed_model_names)} models")
#                     log.info(
#                         f"   │  └─ Models: {', '.join(completed_model_names)}")

#                 # Restore best models to BEST cache
#                 if 'all_task_models' in ckpt_data:
#                     for model_res in ckpt_data['all_task_models']:
#                         if 'model_object' in model_res:
#                             model_name = model_res['model_base_name']
#                             BEST[f"{task}_{domain}_{model_name}"] = model_res['model_object']

#             except Exception as e:
#                 log.warning(
#                     f"   ├─ Failed to load checkpoint {ckpt_path.name}: {e}")

#         if checkpoint_state['completed_tasks']:
#             log.info(
#                 f"   ✅ Resuming from checkpoint - completed tasks: {checkpoint_state['completed_tasks']}")
#             results.extend(checkpoint_state['loaded_results'])

#     # Log pipeline initialization
#     log.info("🚀 Starting MODE A Training Pipeline")
#     log.info(f"   Domain: {domain}")
#     log.info(f"   SMOTE enabled: {apply_smote_flag}")
#     log.info(f"   Silver set size: {len(silver_df):,}")
#     log.info(f"   Gold set size: {len(gold_df):,}")
#     log.info(f"   Feature dimensions: {X_silver.shape}")
#     log.info(f"   Checkpoint directory: {checkpoint_dir}")

#     if checkpoint_state['completed_tasks']:
#         log.info(
#             f"   📂 Resuming with {len(checkpoint_state['loaded_results'])} pre-loaded results")

#     # Show class distribution
#     log.info("\n📊 Class Distribution Analysis:")
#     for task in ("keto", "vegan"):
#         silver_pos = silver_df[f"silver_{task}"].sum()
#         silver_total = len(silver_df)
#         gold_pos = gold_df[f"label_{task}"].sum()
#         gold_total = len(gold_df)

#         log.info(f"   {task.capitalize():>5} - Silver: {silver_pos:,}/{silver_total:,} ({silver_pos/silver_total:.1%}) | "
#                  f"Gold: {gold_pos:,}/{gold_total:,} ({gold_pos/gold_total:.1%})")

#     # Store all models for each task
#     all_task_models = {"keto": [], "vegan": []}

#     # Main training loop
#     task_progress = tqdm(["keto", "vegan"], desc="🔬 Training Tasks",
#                          position=0, leave=True,
#                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

#     for task in task_progress:
#         # Skip if already completed
#         if task in checkpoint_state['completed_tasks']:
#             log.info(
#                 f"\n⏭️  Skipping {task} - already completed (from checkpoint)")

#             # Add the loaded results for this task to all_task_models
#             task_results = [
#                 r for r in checkpoint_state['loaded_results'] if r['task'] == task]
#             all_task_models[task].extend(task_results)

#             task_progress.update(1)
#             continue

#         task_start = time.time()
#         task_progress.set_description(f"🔬 Training {task.capitalize()}")

#         # Extract labels (existing code remains the same)
#         def safe_label_extraction(df, task):
#             """Safely extract labels with NaN handling"""
#             if f"silver_{task}" in df.columns:
#                 label_col = f"silver_{task}"
#             elif f"label_{task}" in df.columns:
#                 label_col = f"label_{task}"
#             else:
#                 log.error(
#                     f"Missing both silver_{task} and label_{task} columns")
#                 default_val = 0 if task == "keto" else 1
#                 return np.full(len(df), default_val, dtype=int)

#             labels = df[label_col].copy()

#             if labels.isna().any():
#                 nan_count = labels.isna().sum()
#                 log.warning(
#                     f"Found {nan_count} NaN values in {task} labels, filling with defaults")
#                 default_val = 0 if task == "keto" else 1
#                 labels = labels.fillna(default_val)

#             try:
#                 labels = labels.astype(int)
#             except ValueError as e:
#                 log.error(f"Cannot convert {task} labels to int: {e}")
#                 labels = pd.to_numeric(labels, errors='coerce').fillna(
#                     0 if task == "keto" else 1).astype(int)

#             return labels.values

#         # Extract labels with validation
#         y_train = safe_label_extraction(silver_df, task)
#         y_true = safe_label_extraction(gold_df, task)
#         log.info(f"\n🎯 Processing {task.upper()} classification:")
#         log.info(
#             f"   Training labels - Positive: {y_train.sum():,} ({y_train.mean():.1%})")
#         log.info(
#             f"   Test labels - Positive: {y_true.sum():,} ({y_true.mean():.1%})")

#         # Handle class imbalance (existing SMOTE code)
#         if apply_smote_flag:
#             # ... (existing SMOTE code remains the same)
#             X_train = X_silver  # Simplified for this example
#         else:
#             X_train = X_silver

#         # Build and train models
#         models = build_models(task, domain)

#         # Filter out Rule model for image domain
#         if domain == "image":
#             models = {k: v for k, v in models.items() if k != "Rule"}

#         # Check if any models were already completed in a partial checkpoint
#         completed_for_task = checkpoint_state['completed_models'].get(task, [])
#         if completed_for_task:
#             log.info(
#                 f"   📂 Found {len(completed_for_task)} already completed models for {task}")
#             # Filter out completed models
#             models = {k: v for k, v in models.items()
#                       if f"{k}_{domain.upper()}" not in completed_for_task}

#         log.info(f"   🤖 Training {len(models)} models: {list(models.keys())}")

#         # Initialize task-specific best tracking
#         best_f1 = -1.0
#         best_res = None
#         model_results = []

#         # Model training progress
#         model_progress = tqdm(models.items(), desc="   ├─ Training Models",
#                               position=1, leave=False,
#                               bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

#         # ... (rest of the model training code remains the same until the checkpoint saving)

#         # After training all models for this task, save checkpoint
#         checkpoint_path = checkpoint_dir / f"checkpoint_{task}_{domain}.pkl"
#         checkpoint_data = {
#             'task': task,
#             'domain': domain,
#             'best_result': best_res,
#             'all_results': model_results,
#             'all_task_models': all_task_models[task],
#             'timestamp': datetime.now().isoformat(),
#             'task_time': time.time() - task_start,
#             'feature_shape': X_silver.shape,
#             'n_models_trained': len(model_results),
#             # Track model names
#             'completed_models': [r['model'] for r in model_results],
#         }

#         try:
#             with open(checkpoint_path, 'wb') as f:
#                 pickle.dump(checkpoint_data, f)
#             log.info(f"   💾 Checkpoint saved: {checkpoint_path.name}")

#             # Also update pipeline state
#             PIPELINE_STATE.save_checkpoint(f"{task}_{domain}", checkpoint_data)

#         except Exception as e:
#             log.error(f"   ❌ Failed to save checkpoint: {e}")

#         # Update progress
#         task_progress.set_postfix({
#             'Best': best_res['model'] if best_res else 'N/A',
#             'F1': f"{best_res['F1']:.3f}" if best_res else '0.000',
#             'Time': f"{time.time() - task_start:.1f}s"
#         })

#     # Pipeline completion
#     pipeline_time = time.time() - pipeline_start

#     # Store all task models in results metadata
#     results_metadata = {
#         'all_task_models': all_task_models,
#         'pipeline_time': pipeline_time,
#         'domain': domain,
#         'silver_size': len(silver_df),
#         'gold_size': len(gold_df),
#         'feature_dimensions': X_silver.shape,
#         'smote_applied': apply_smote_flag,
#         'timestamp': datetime.now().isoformat(),
#         'checkpoint_dir': str(checkpoint_dir)
#     }

#     # Export results to CSV
#     export_results_to_csv(results, results_metadata, domain)

#     # Pipeline completion
#     log.info(f"\n🏁 MODE A PIPELINE COMPLETE:")
#     log.info(f"   ├─ Total Time: {pipeline_time:.1f}s")
#     log.info(f"   ├─ Tasks Completed: 2")
#     log.info(f"   ├─ Models Trained: {len(results)}")
#     log.info(f"   ├─ Domain: {domain}")
#     log.info(f"   └─ Checkpoints saved: {checkpoint_dir}")

#     # Summary table - show ALL results, not just best per task
#     log.info(f"\n📊 ALL RESULTS:")
#     for i, res in enumerate(results, 1):
#         log.info(f"   {i:2d}. {res['task'].upper():>5} | {res['model']:>15} | "
#                  f"F1={res['F1']:.3f} | ACC={res['ACC']:.3f} | "
#                  f"Time={res.get('training_time', 0):.1f}s")

#     # Display formatted table
#     table("MODE A (silver → gold)", results)

#     # Return results with metadata attached
#     for res in results:
#         res['_metadata'] = results_metadata

#     return results
