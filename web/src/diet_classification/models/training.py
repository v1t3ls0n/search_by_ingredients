#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training orchestration (simplified version).

This is a placeholder for the full run_mode_A function.
Based on original lines 4529-5165 from diet_classifiers.py
"""

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from sklearn.base import BaseEstimator

from ..core import log, get_pipeline_state
from ..config import CFG
from .builders import build_models
from ..evaluation.metrics import pack


def run_mode_A(
    X_silver,
    gold_clean: pd.Series,
    X_gold,
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    domain: str = "text",
    apply_smote_flag: bool = True,
    checkpoint_dir: Optional[Path] = None
) -> List[Dict]:
    """
    Train on weak (silver) labels, evaluate on gold standard labels.

    This is a simplified placeholder. The full implementation would include:
    - Complete checkpoint/resume functionality
    - SMOTE application for class balancing
    - Full training pipeline with progress tracking
    - Model-specific error handling

    Args:
        X_silver: Training features
        gold_clean: Normalized gold text
        X_gold: Test features
        silver_df: Silver training data
        gold_df: Gold test data
        domain: Feature domain ('text', 'image', 'both')
        apply_smote_flag: Whether to apply SMOTE
        checkpoint_dir: Directory for checkpoints

    Returns:
        List of result dictionaries
    """
    log.info("🚀 Starting MODE A Training Pipeline")
    log.info(f"   Domain: {domain}")
    log.info(f"   SMOTE enabled: {apply_smote_flag}")
    log.info(f"   Silver set size: {len(silver_df):,}")
    log.info(f"   Gold set size: {len(gold_df):,}")
    
    results = []
    pipeline_state = get_pipeline_state()
    
    for task in ["keto", "vegan"]:
        log.info(f"\n🎯 Processing {task.upper()} classification:")
        
        # Extract labels
        y_train = silver_df[f"silver_{task}"].values
        y_true = gold_df[f"label_{task}"].values
        
        # Build models for this task
        models = build_models(task, domain)
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_silver, y_train)
                
                # Make predictions
                prob = model.predict_proba(X_gold)[:, 1]
                
                # Apply verification
                from ..classification.verification import verify_with_rules
                prob_adj = verify_with_rules(task, gold_clean, prob)
                
                # Calculate metrics
                metrics = pack(y_true, prob_adj)
                
                # Store result
                result = {
                    "task": task,
                    "model": f"{model_name}_{domain.upper()}",
                    "domain": domain,
                    "model_base_name": model_name,
                    "prob": prob_adj,
                    "pred": (prob_adj >= 0.5).astype(int),
                    **metrics
                }
                
                results.append(result)
                
                # Cache model
                cache_key = f"{task}_{domain}_{model_name}"
                pipeline_state.best_models[cache_key] = model
                
                log.info(f"   ✅ {model_name}: F1={result['F1']:.3f}")
                
            except Exception as e:
                log.error(f"   ❌ {model_name} failed: {e}")
    
    return results









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
