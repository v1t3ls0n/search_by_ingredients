#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline evaluation functions for ground truth testing.

This module provides evaluation capabilities for testing trained models
against labeled ground truth data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from ..core import log, get_pipeline_state
from ..config import CFG
from ..classification.keto import is_keto
from ..classification.vegan import is_vegan
from ..data.preprocessing import normalise
from ..evaluation.metrics import pack, table


def evaluate_ground_truth(ground_truth_path: str) -> None:
    """
    Evaluate trained models against ground truth labeled data.
    
    Args:
        ground_truth_path: Path to CSV file with ground truth labels
    """
    log.info("📊 GROUND TRUTH EVALUATION")
    
    try:
        # Load ground truth data
        log.info(f"   ├─ Loading ground truth: {ground_truth_path}")
        gt_df = pd.read_csv(ground_truth_path)
        
        # Validate required columns
        required_cols = ['ingredients']
        label_cols = [col for col in gt_df.columns if 'keto' in col.lower() or 'vegan' in col.lower()]
        
        if not required_cols[0] in gt_df.columns:
            log.error(f"   ❌ Missing required column: {required_cols[0]}")
            return
            
        if not label_cols:
            log.error(f"   ❌ No label columns found (looking for 'keto' or 'vegan' in column names)")
            return
            
        log.info(f"   ├─ Dataset size: {len(gt_df)} samples")
        log.info(f"   ├─ Available label columns: {label_cols}")
        
        # Prepare data
        gt_df['clean'] = gt_df['ingredients'].fillna("").map(normalise)
        
        # Extract labels (look for any column containing keto/vegan)
        keto_col = next((col for col in gt_df.columns if 'keto' in col.lower()), None)
        vegan_col = next((col for col in gt_df.columns if 'vegan' in col.lower()), None)
        
        results = []
        
        # Evaluate keto classification
        if keto_col is not None:
            log.info(f"   🥑 Evaluating keto classification using column: {keto_col}")
            
            y_true_keto = gt_df[keto_col].astype(int)
            y_pred_keto = []
            
            for ingredients in gt_df['ingredients']:
                try:
                    pred = is_keto(ingredients)
                    y_pred_keto.append(int(pred))
                except Exception as e:
                    log.warning(f"   ⚠️  Keto prediction failed for '{ingredients[:50]}...': {e}")
                    y_pred_keto.append(0)  # Default to non-keto on error
            
            # Calculate metrics
            y_pred_keto_probs = [float(p) for p in y_pred_keto]  # Convert to probabilities
            keto_metrics = pack(y_true_keto, y_pred_keto_probs)
            
            result = {
                'task': 'keto',
                'model': 'Production_API',
                **keto_metrics
            }
            results.append(result)
            
            # Log summary
            correct = sum(1 for t, p in zip(y_true_keto, y_pred_keto) if t == p)
            accuracy = correct / len(y_true_keto)
            log.info(f"      ├─ Accuracy: {accuracy:.3f} ({correct}/{len(y_true_keto)})")
            log.info(f"      ├─ F1 Score: {keto_metrics['F1']:.3f}")
            log.info(f"      └─ Precision: {keto_metrics['PREC']:.3f}, Recall: {keto_metrics['REC']:.3f}")
        
        # Evaluate vegan classification  
        if vegan_col is not None:
            log.info(f"   🌱 Evaluating vegan classification using column: {vegan_col}")
            
            y_true_vegan = gt_df[vegan_col].astype(int)
            y_pred_vegan = []
            
            for ingredients in gt_df['ingredients']:
                try:
                    pred = is_vegan(ingredients)
                    y_pred_vegan.append(int(pred))
                except Exception as e:
                    log.warning(f"   ⚠️  Vegan prediction failed for '{ingredients[:50]}...': {e}")
                    y_pred_vegan.append(1)  # Default to vegan on error
            
            # Calculate metrics
            y_pred_vegan_probs = [float(p) for p in y_pred_vegan]
            vegan_metrics = pack(y_true_vegan, y_pred_vegan_probs)
            
            result = {
                'task': 'vegan', 
                'model': 'Production_API',
                **vegan_metrics
            }
            results.append(result)
            
            # Log summary
            correct = sum(1 for t, p in zip(y_true_vegan, y_pred_vegan) if t == p)
            accuracy = correct / len(y_true_vegan)
            log.info(f"      ├─ Accuracy: {accuracy:.3f} ({correct}/{len(y_true_vegan)})")
            log.info(f"      ├─ F1 Score: {vegan_metrics['F1']:.3f}")
            log.info(f"      └─ Precision: {vegan_metrics['PREC']:.3f}, Recall: {vegan_metrics['REC']:.3f}")
        
        # Display results table
        if results:
            table("Ground Truth Evaluation", results)
            
            # Save detailed results
            output_dir = CFG.artifacts_dir / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions vs ground truth
            eval_results = gt_df.copy()
            if keto_col and len(y_pred_keto) == len(gt_df):
                eval_results['predicted_keto'] = y_pred_keto
                eval_results['keto_correct'] = eval_results[keto_col] == eval_results['predicted_keto']
            
            if vegan_col and len(y_pred_vegan) == len(gt_df):
                eval_results['predicted_vegan'] = y_pred_vegan  
                eval_results['vegan_correct'] = eval_results[vegan_col] == eval_results['predicted_vegan']
            
            output_path = output_dir / "ground_truth_evaluation.csv"
            eval_results.to_csv(output_path, index=False)
            log.info(f"   💾 Detailed results saved to: {output_path}")
            
            # Summary metrics
            summary_path = output_dir / "evaluation_summary.csv"
            pd.DataFrame(results).to_csv(summary_path, index=False)
            log.info(f"   📊 Summary metrics saved to: {summary_path}")
            
        else:
            log.warning(f"   ⚠️  No evaluations performed - check label columns")
            
        log.info(f"   ✅ Ground truth evaluation complete")
        
    except FileNotFoundError:
        log.error(f"   ❌ Ground truth file not found: {ground_truth_path}")
    except Exception as e:
        log.error(f"   ❌ Evaluation failed: {e}")
        import traceback
        log.debug(f"Full traceback:\n{traceback.format_exc()}")


def evaluate_models_on_dataset(dataset_path: str, model_names: Optional[list] = None) -> None:
    """
    Evaluate specific trained models against a dataset.
    
    This function loads trained models and evaluates them against labeled data,
    useful for comparing different model configurations.
    
    Args:
        dataset_path: Path to labeled dataset
        model_names: List of model names to evaluate (if None, evaluates all available)
    """
    log.info("🔬 MODEL EVALUATION ON DATASET")
    
    pipeline_state = get_pipeline_state()
    pipeline_state.ensure_pipeline_initialized()
    
    if not pipeline_state.initialized:
        log.error("   ❌ No trained models available for evaluation")
        log.info("   ℹ️  Run training first with --train")
        return
    
    log.info(f"   ├─ Available models: {list(pipeline_state.models.keys())}")
    log.info(f"   └─ Evaluating on: {dataset_path}")
    
    # Implementation would load models and run detailed evaluation
    # For now, fall back to API-based evaluation
    log.info("   ⏭️  Falling back to API-based evaluation...")
    evaluate_ground_truth(dataset_path)