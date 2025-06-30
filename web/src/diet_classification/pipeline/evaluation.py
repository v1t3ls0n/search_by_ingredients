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
from numpy import np

from ..core import log, get_pipeline_state
from ..config import CFG
from ..classification.keto import is_keto
from ..classification.vegan import is_vegan
from ..data.preprocessing import normalise
from ..evaluation.metrics import pack, table


def evaluate_ground_truth(ground_truth_path: str, mode: str = 'text') -> None:
    """
    Evaluate trained models against ground truth labeled data.
    
    Args:
        ground_truth_path: Path to CSV file with ground truth labels
        mode: Feature mode to use ('text', 'image', or 'both')
    """
    log.info("📊 GROUND TRUTH EVALUATION")
    log.info(f"   ├─ Mode: {mode}")
    
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
            log.info(f"   ├─ Using mode: {mode}")
            
            y_true_keto = gt_df[keto_col].astype(int)
            y_pred_keto = []
            y_pred_keto_probs = []
            
            # Track prediction times
            prediction_times = []
            import time
            
            for idx, ingredients in enumerate(gt_df['ingredients']):
                try:
                    start_time = time.time()
                    # Pass mode parameter to is_keto
                    pred = is_keto(ingredients, mode=mode)
                    prediction_time = time.time() - start_time
                    prediction_times.append(prediction_time)
                    
                    y_pred_keto.append(int(pred))
                    y_pred_keto_probs.append(float(pred))
                except Exception as e:
                    log.warning(f"   ⚠️  Keto prediction failed for row {idx} '{ingredients[:50]}...': {e}")
                    y_pred_keto.append(0)  # Default to non-keto on error
                    y_pred_keto_probs.append(0.0)
                    prediction_times.append(0.0)
            
            # Calculate metrics
            keto_metrics = pack(y_true_keto, y_pred_keto_probs)
            
            result = {
                'task': 'keto',
                'model': f'Production_API_{mode}',
                'mode': mode,
                'avg_prediction_time': np.mean(prediction_times) if prediction_times else 0.0,
                **keto_metrics
            }
            results.append(result)
            
            # Log summary
            correct = sum(1 for t, p in zip(y_true_keto, y_pred_keto) if t == p)
            accuracy = correct / len(y_true_keto)
            log.info(f"      ├─ Accuracy: {accuracy:.3f} ({correct}/{len(y_true_keto)})")
            log.info(f"      ├─ F1 Score: {keto_metrics['F1']:.3f}")
            log.info(f"      ├─ Precision: {keto_metrics['PREC']:.3f}, Recall: {keto_metrics['REC']:.3f}")
            log.info(f"      └─ Avg prediction time: {np.mean(prediction_times):.3f}s")
        
        # Evaluate vegan classification  
        if vegan_col is not None:
            log.info(f"   🌱 Evaluating vegan classification using column: {vegan_col}")
            log.info(f"   ├─ Using mode: {mode}")
            
            y_true_vegan = gt_df[vegan_col].astype(int)
            y_pred_vegan = []
            y_pred_vegan_probs = []
            
            # Track prediction times
            prediction_times = []
            import time
            
            for idx, ingredients in enumerate(gt_df['ingredients']):
                try:
                    start_time = time.time()
                    # Pass mode parameter to is_vegan
                    pred = is_vegan(ingredients, mode=mode)
                    prediction_time = time.time() - start_time
                    prediction_times.append(prediction_time)
                    
                    y_pred_vegan.append(int(pred))
                    y_pred_vegan_probs.append(float(pred))
                except Exception as e:
                    log.warning(f"   ⚠️  Vegan prediction failed for row {idx} '{ingredients[:50]}...': {e}")
                    y_pred_vegan.append(1)  # Default to vegan on error
                    y_pred_vegan_probs.append(1.0)
                    prediction_times.append(0.0)
            
            # Calculate metrics
            vegan_metrics = pack(y_true_vegan, y_pred_vegan_probs)
            
            result = {
                'task': 'vegan', 
                'model': f'Production_API_{mode}',
                'mode': mode,
                'avg_prediction_time': np.mean(prediction_times) if prediction_times else 0.0,
                **vegan_metrics
            }
            results.append(result)
            
            # Log summary
            correct = sum(1 for t, p in zip(y_true_vegan, y_pred_vegan) if t == p)
            accuracy = correct / len(y_true_vegan)
            log.info(f"      ├─ Accuracy: {accuracy:.3f} ({correct}/{len(y_true_vegan)})")
            log.info(f"      ├─ F1 Score: {vegan_metrics['F1']:.3f}")
            log.info(f"      ├─ Precision: {vegan_metrics['PREC']:.3f}, Recall: {vegan_metrics['REC']:.3f}")
            log.info(f"      └─ Avg prediction time: {np.mean(prediction_times):.3f}s")
        
        # Display results table
        if results:
            table(f"Ground Truth Evaluation (Mode: {mode})", results)
            
            # Save detailed results
            output_dir = CFG.artifacts_dir / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions vs ground truth
            eval_results = gt_df.copy()
            eval_results['evaluation_mode'] = mode
            
            if keto_col and len(y_pred_keto) == len(gt_df):
                eval_results['predicted_keto'] = y_pred_keto
                eval_results['keto_probability'] = y_pred_keto_probs
                eval_results['keto_correct'] = eval_results[keto_col] == eval_results['predicted_keto']
                
                # Add error analysis
                eval_results['keto_error_type'] = 'correct'
                eval_results.loc[(eval_results[keto_col] == 1) & (eval_results['predicted_keto'] == 0), 'keto_error_type'] = 'false_negative'
                eval_results.loc[(eval_results[keto_col] == 0) & (eval_results['predicted_keto'] == 1), 'keto_error_type'] = 'false_positive'
            
            if vegan_col and len(y_pred_vegan) == len(gt_df):
                eval_results['predicted_vegan'] = y_pred_vegan
                eval_results['vegan_probability'] = y_pred_vegan_probs
                eval_results['vegan_correct'] = eval_results[vegan_col] == eval_results['predicted_vegan']
                
                # Add error analysis
                eval_results['vegan_error_type'] = 'correct'
                eval_results.loc[(eval_results[vegan_col] == 1) & (eval_results['predicted_vegan'] == 0), 'vegan_error_type'] = 'false_negative'
                eval_results.loc[(eval_results[vegan_col] == 0) & (eval_results['predicted_vegan'] == 1), 'vegan_error_type'] = 'false_positive'
            
            # Save with mode in filename
            output_path = output_dir / f"ground_truth_evaluation_{mode}.csv"
            eval_results.to_csv(output_path, index=False)
            log.info(f"   💾 Detailed results saved to: {output_path}")
            
            # Save error analysis
            if keto_col:
                keto_errors = eval_results[eval_results['keto_error_type'] != 'correct'][['ingredients', 'clean', keto_col, 'predicted_keto', 'keto_error_type']]
                if not keto_errors.empty:
                    error_path = output_dir / f"keto_errors_{mode}.csv"
                    keto_errors.to_csv(error_path, index=False)
                    log.info(f"   📝 Keto errors saved to: {error_path}")
            
            if vegan_col:
                vegan_errors = eval_results[eval_results['vegan_error_type'] != 'correct'][['ingredients', 'clean', vegan_col, 'predicted_vegan', 'vegan_error_type']]
                if not vegan_errors.empty:
                    error_path = output_dir / f"vegan_errors_{mode}.csv"
                    vegan_errors.to_csv(error_path, index=False)
                    log.info(f"   📝 Vegan errors saved to: {error_path}")
            
            # Summary metrics
            summary_path = output_dir / f"evaluation_summary_{mode}.csv"
            pd.DataFrame(results).to_csv(summary_path, index=False)
            log.info(f"   📊 Summary metrics saved to: {summary_path}")
            
            # Create a comprehensive summary report
            summary_report = []
            summary_report.append(f"GROUND TRUTH EVALUATION REPORT")
            summary_report.append(f"Mode: {mode}")
            summary_report.append(f"Dataset: {ground_truth_path}")
            summary_report.append(f"Total samples: {len(gt_df)}")
            summary_report.append("")
            
            for result in results:
                task = result['task']
                summary_report.append(f"{task.upper()} CLASSIFICATION:")
                summary_report.append(f"  Accuracy: {result['ACC']:.3f}")
                summary_report.append(f"  Precision: {result['PREC']:.3f}")
                summary_report.append(f"  Recall: {result['REC']:.3f}")
                summary_report.append(f"  F1 Score: {result['F1']:.3f}")
                summary_report.append(f"  ROC AUC: {result['ROC']:.3f}")
                summary_report.append(f"  PR AUC: {result['PR']:.3f}")
                summary_report.append(f"  Avg prediction time: {result['avg_prediction_time']:.3f}s")
                summary_report.append("")
            
            report_path = output_dir / f"evaluation_report_{mode}.txt"
            with open(report_path, 'w') as f:
                f.write('\n'.join(summary_report))
            log.info(f"   📄 Summary report saved to: {report_path}")
            
        else:
            log.warning(f"   ⚠️  No evaluations performed - check label columns")
            
        log.info(f"   ✅ Ground truth evaluation complete (mode: {mode})")
        
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