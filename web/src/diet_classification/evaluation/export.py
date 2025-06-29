#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export module for saving evaluation results.

Provides functions for exporting results to various formats:
- CSV files for metrics and predictions
- JSON for metadata
- Model artifacts
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from ..core import log
from ..config import CFG


def export_results_to_csv(
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    domain: str,
    output_dir: Optional[Path] = None
):
    """
    Export training results and metadata to CSV files.
    
    Creates multiple CSV files:
    - model_metrics_{domain}_{timestamp}.csv: All model performance metrics
    - best_models_{domain}_{timestamp}.csv: Best model per task
    - training_metadata_{domain}_{timestamp}.csv: Training run metadata
    - model_comparison_{domain}_{timestamp}.csv: Side-by-side comparison
    
    Args:
        results: List of result dictionaries
        metadata: Training metadata
        domain: Feature domain ('text', 'image', 'both')
        output_dir: Output directory (defaults to artifacts/metrics)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export all model metrics
    metrics_data = []
    for res in results:
        metrics_data.append({
            'timestamp': timestamp,
            'domain': domain,
            'task': res['task'],
            'model': res['model'],
            'model_base_name': res.get('model_base_name', res['model'].split('_')[0]),
            'accuracy': res.get('ACC', 0),
            'precision': res.get('PREC', 0),
            'recall': res.get('REC', 0),
            'f1_score': res.get('F1', 0),
            'roc_auc': res.get('ROC', 0),
            'pr_auc': res.get('PR', 0),
            'training_time_seconds': res.get('training_time', 0),
            'feature_domain': res.get('domain', domain),
            'n_samples_train': metadata.get('silver_size', 0),
            'n_samples_test': metadata.get('gold_size', 0),
            'feature_dimensions': str(metadata.get('feature_dimensions', ''))
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = output_dir / f"model_metrics_{domain}_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"   📊 Saved model metrics to {metrics_path}")
    
    # 2. Export best models per task
    best_models_data = []
    for task in ['keto', 'vegan']:
        task_results = [r for r in results if r['task'] == task]
        if task_results:
            best_model = max(task_results, key=lambda x: x.get('F1', 0))
            best_models_data.append({
                'timestamp': timestamp,
                'domain': domain,
                'task': task,
                'best_model': best_model['model'],
                'f1_score': best_model.get('F1', 0),
                'accuracy': best_model.get('ACC', 0),
                'precision': best_model.get('PREC', 0),
                'recall': best_model.get('REC', 0),
                'roc_auc': best_model.get('ROC', 0),
                'pr_auc': best_model.get('PR', 0),
                'training_time': best_model.get('training_time', 0)
            })
    
    if best_models_data:
        best_df = pd.DataFrame(best_models_data)
        best_path = output_dir / f"best_models_{domain}_{timestamp}.csv"
        best_df.to_csv(best_path, index=False)
        log.info(f"   🏆 Saved best models to {best_path}")
    
    # 3. Export training metadata
    metadata_data = [{
        'timestamp': timestamp,
        'domain': domain,
        'pipeline_time_seconds': metadata.get('pipeline_time', 0),
        'silver_size': metadata.get('silver_size', 0),
        'gold_size': metadata.get('gold_size', 0),
        'feature_dimensions': str(metadata.get('feature_dimensions', '')),
        'smote_applied': metadata.get('smote_applied', False),
        'n_models_trained': len(results),
        'training_timestamp': metadata.get('timestamp', timestamp)
    }]
    
    metadata_df = pd.DataFrame(metadata_data)
    metadata_path = output_dir / f"training_metadata_{domain}_{timestamp}.csv"
    metadata_df.to_csv(metadata_path, index=False)
    log.info(f"   📋 Saved training metadata to {metadata_path}")
    
    # 4. Create model comparison matrix
    comparison_data = []
    model_types = sorted(set(r.get('model_base_name', r['model'].split('_')[0]) for r in results))
    
    for model_type in model_types:
        row = {'model_type': model_type}
        for task in ['keto', 'vegan']:
            task_model = next((r for r in results
                              if r['task'] == task and
                              r.get('model_base_name', r['model'].split('_')[0]) == model_type),
                              None)
            if task_model:
                row[f'{task}_f1'] = task_model.get('F1', 0)
                row[f'{task}_acc'] = task_model.get('ACC', 0)
                row[f'{task}_time'] = task_model.get('training_time', 0)
            else:
                row[f'{task}_f1'] = None
                row[f'{task}_acc'] = None
                row[f'{task}_time'] = None
        comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = output_dir / f"model_comparison_{domain}_{timestamp}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        log.info(f"   📈 Saved model comparison to {comparison_path}")


def export_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    texts: pd.Series,
    task: str,
    model_name: str,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Export detailed predictions for error analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        texts: Original text data
        task: Task name
        model_name: Model name
        output_dir: Output directory
        
    Returns:
        Path to saved predictions file
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    predictions_df = pd.DataFrame({
        'text': texts.values if hasattr(texts, 'values') else texts,
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_prob,
        'correct': y_true == y_pred,
        'confidence': np.abs(y_prob - 0.5) * 2,  # Distance from 0.5
        'error_type': np.where(
            y_true == y_pred, 'correct',
            np.where(y_pred == 1, 'false_positive', 'false_negative')
        )
    })
    
    # Sort by confidence for interesting cases
    predictions_df = predictions_df.sort_values('confidence', ascending=False)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = output_dir / f"predictions_{task}_{model_name}_{timestamp}.csv"
    predictions_df.to_csv(pred_path, index=False)
    
    log.info(f"   💾 Saved predictions to {pred_path}")
    
    # Also save error analysis summary
    error_summary = predictions_df.groupby('error_type').agg({
        'text': 'count',
        'confidence': ['mean', 'std']
    })
    
    summary_path = output_dir / f"error_summary_{task}_{model_name}_{timestamp}.csv"
    error_summary.to_csv(summary_path)
    
    return pred_path


def log_false_predictions(
    task: str,
    texts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Optional[Path] = None,
    max_examples: int = 50
):
    """
    Log false positive and false negative predictions for analysis.
    
    Args:
        task: Task name
        texts: Original text data
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name
        output_dir: Output directory
        max_examples: Maximum examples to save per error type
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # False Positives
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    
    if len(fp_indices) > 0:
        fp_examples = []
        for idx in fp_indices[:max_examples]:
            fp_examples.append({
                'index': idx,
                'text': texts.iloc[idx] if hasattr(texts, 'iloc') else texts[idx],
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx]),
                'error_type': 'false_positive'
            })
        
        fp_df = pd.DataFrame(fp_examples)
        fp_path = output_dir / f"false_positives_{task}_{model_name}.csv"
        fp_df.to_csv(fp_path, index=False)
        log.info(f"   📝 Logged {len(fp_df)} false positives to {fp_path}")
    
    # False Negatives
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    
    if len(fn_indices) > 0:
        fn_examples = []
        for idx in fn_indices[:max_examples]:
            fn_examples.append({
                'index': idx,
                'text': texts.iloc[idx] if hasattr(texts, 'iloc') else texts[idx],
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx]),
                'error_type': 'false_negative'
            })
        
        fn_df = pd.DataFrame(fn_examples)
        fn_path = output_dir / f"false_negatives_{task}_{model_name}.csv"
        fn_df.to_csv(fn_path, index=False)
        log.info(f"   📝 Logged {len(fn_df)} false negatives to {fn_path}")


def save_model_artifacts(
    results: List[Dict[str, Any]],
    domain: str,
    output_dir: Optional[Path] = None
):
    """
    Save model objects and related artifacts.
    
    Args:
        results: List of results containing model objects
        domain: Feature domain
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "models" / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual models
    saved_models = {}
    
    for result in results:
        if 'model_object' in result:
            model_name = result['model']
            model_obj = result['model_object']
            
            # Save with joblib (compressed)
            model_path = output_dir / f"{model_name}_{timestamp}.pkl.gz"
            joblib.dump(model_obj, model_path, compress=3)
            
            saved_models[model_name] = {
                'path': str(model_path),
                'task': result['task'],
                'f1_score': result.get('F1', 0),
                'domain': domain
            }
    
    # Save model index
    if saved_models:
        index_path = output_dir / f"model_index_{timestamp}.json"
        with open(index_path, 'w') as f:
            json.dump(saved_models, f, indent=2)
        
        log.info(f"   💾 Saved {len(saved_models)} models to {output_dir}")


def aggregate_results_across_domains(
    all_results: Dict[str, List[Dict]],
    output_dir: Optional[Path] = None
):
    """
    Aggregate results across different domains for final comparison.
    
    Args:
        all_results: Dictionary mapping domain -> list of results
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive comparison
    all_metrics = []
    
    for domain, results in all_results.items():
        for res in results:
            all_metrics.append({
                'domain': domain,
                'task': res.get('task', 'unknown'),
                'model': res.get('model', 'unknown'),
                'model_base_name': res.get('model_base_name', res.get('model', '').split('_')[0]),
                'f1_score': res.get('F1', 0),
                'accuracy': res.get('ACC', 0),
                'precision': res.get('PREC', 0),
                'recall': res.get('REC', 0),
                'roc_auc': res.get('ROC', 0),
                'pr_auc': res.get('PR', 0),
                'training_time': res.get('training_time', 0)
            })
    
    if all_metrics:
        # 1. All metrics across domains
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_path = output_dir / f"all_domain_metrics_{timestamp}.csv"
        all_metrics_df.to_csv(all_metrics_path, index=False)
        log.info(f"\n📊 CROSS-DOMAIN METRICS EXPORT:")
        log.info(f"   ├─ Saved all metrics to {all_metrics_path}")
        
        # 2. Best model per task across all domains
        best_across_domains = []
        for task in ['keto', 'vegan']:
            task_models = [m for m in all_metrics if m['task'] == task]
            if task_models:
                best = max(task_models, key=lambda x: x['f1_score'])
                best_across_domains.append(best)
        
        if best_across_domains:
            best_df = pd.DataFrame(best_across_domains)
            best_path = output_dir / f"best_models_all_domains_{timestamp}.csv"
            best_df.to_csv(best_path, index=False)
            log.info(f"   ├─ Saved best models to {best_path}")
        
        # 3. Domain comparison summary
        domain_summary = []
        for domain in all_results.keys():
            domain_metrics = [m for m in all_metrics if m['domain'] == domain]
            if domain_metrics:
                domain_summary.append({
                    'domain': domain,
                    'n_models': len(domain_metrics),
                    'avg_f1_keto': np.mean([m['f1_score'] for m in domain_metrics if m['task'] == 'keto'] or [0]),
                    'avg_f1_vegan': np.mean([m['f1_score'] for m in domain_metrics if m['task'] == 'vegan'] or [0]),
                    'best_f1_keto': max([m['f1_score'] for m in domain_metrics if m['task'] == 'keto'] or [0], default=0),
                    'best_f1_vegan': max([m['f1_score'] for m in domain_metrics if m['task'] == 'vegan'] or [0], default=0),
                    'avg_training_time': np.mean([m['training_time'] for m in domain_metrics])
                })
        
        if domain_summary:
            domain_df = pd.DataFrame(domain_summary)
            domain_path = output_dir / f"domain_comparison_{timestamp}.csv"
            domain_df.to_csv(domain_path, index=False)
            log.info(f"   └─ Saved domain comparison to {domain_path}")


def export_ensemble_metrics(
    ensemble_results: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
):
    """
    Export specific metrics for ensemble models.
    
    Args:
        ensemble_results: List of ensemble results
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "metrics" / "ensembles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ensemble_data = []
    
    for res in ensemble_results:
        if any(key in str(res.get('model', '')).lower() for key in ['ens', 'best', 'vote', 'stack', 'blend']):
            ensemble_data.append({
                'timestamp': timestamp,
                'task': res.get('task', 'unknown'),
                'ensemble_type': res.get('ensemble_type', 'unknown'),
                'model_name': res.get('model', 'unknown'),
                'f1_score': res.get('F1', 0),
                'accuracy': res.get('ACC', 0),
                'precision': res.get('PREC', 0),
                'recall': res.get('REC', 0),
                'component_models': json.dumps(res.get('models_used', [])),
                'n_models': res.get('n_models', len(res.get('models_used', []))),
                'alpha': res.get('alpha'),
                'preparation_time': res.get('total_time', res.get('training_time', 0))
            })
    
    if ensemble_data:
        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_path = output_dir / f"ensemble_metrics_{timestamp}.csv"
        ensemble_df.to_csv(ensemble_path, index=False)
        log.info(f"   🎭 Saved ensemble metrics to {ensemble_path}")


def export_eval_plots(
    results: List[Dict[str, Any]],
    gold_df: pd.DataFrame,
    output_dir: Optional[Path] = None
):
    """
    Export evaluation plots including confusion matrices and performance comparisons.
    
    Args:
        results: List of results
        gold_df: Gold standard DataFrame
        output_dir: Output directory
    """
    from .visualization import (
        plot_confusion_matrix,
        plot_model_comparison,
        create_performance_report
    )
    
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    
    log.info("   📊 Generating evaluation plots...")
    
    # Create performance report with all visualizations
    create_performance_report(results, output_dir)