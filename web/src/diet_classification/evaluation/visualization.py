#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module for model evaluation.

Provides functions for creating plots and visualizations including:
- Confusion matrices
- ROC curves
- Precision-recall curves
- Performance comparisons
- Feature importance plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

from ..core import log
from ..config import CFG


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    model_name: str,
    output_dir: Optional[Path] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6)
) -> Path:
    """
    Plot confusion matrix with detailed annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task: Task name ('keto' or 'vegan')
        model_name: Model name for title
        output_dir: Directory to save plot
        normalize: Whether to normalize values
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        xticklabels=[f'Not {task.capitalize()}', task.capitalize()],
        yticklabels=[f'Not {task.capitalize()}', task.capitalize()]
    )
    
    # Add raw counts as text
    if normalize:
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                        ha='center', va='center', fontsize=8, color='gray')
    
    plt.title(f'Confusion Matrix - {model_name} ({task.capitalize()})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    plt.text(1.02, 0.5, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"confusion_matrix_{task}_{model_name.replace('/', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved confusion matrix to {plot_path}")
    return plot_path


def plot_roc_curves(
    results: List[Dict[str, Any]],
    task: str,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Path:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: List of result dictionaries with 'prob' and true labels
        task: Task name
        output_dir: Directory to save plot
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Filter results for task
    task_results = [r for r in results if r.get('task') == task and 'prob' in r]
    
    # Plot each model
    for result in task_results:
        model_name = result['model']
        y_true = result.get('y_true', np.ones(len(result['prob'])))  # Placeholder if not available
        y_scores = result['prob']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {task.capitalize()} Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / f"roc_curves_{task}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved ROC curves to {plot_path}")
    return plot_path


def plot_precision_recall_curves(
    results: List[Dict[str, Any]],
    task: str,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Path:
    """
    Plot precision-recall curves for multiple models.
    
    Args:
        results: List of result dictionaries
        task: Task name
        output_dir: Directory to save plot
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Filter results
    task_results = [r for r in results if r.get('task') == task and 'prob' in r]
    
    for result in task_results:
        model_name = result['model']
        y_true = result.get('y_true', np.ones(len(result['prob'])))
        y_scores = result['prob']
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Plot
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves - {task.capitalize()} Classification', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / f"pr_curves_{task}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved PR curves to {plot_path}")
    return plot_path


def plot_model_comparison(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Path:
    """
    Create bar plots comparing models across metrics.
    
    Args:
        results: List of result dictionaries
        metrics: Metrics to compare (default: F1, ACC, PREC, REC)
        output_dir: Directory to save plot
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if metrics is None:
        metrics = ['F1', 'ACC', 'PREC', 'REC']
    
    # Prepare data
    data = []
    for result in results:
        for metric in metrics:
            if metric in result:
                data.append({
                    'Model': result['model'],
                    'Task': result['task'],
                    'Metric': metric,
                    'Score': result[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create subplots for each task
    tasks = df['Task'].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=figsize, sharey=True)
    
    if len(tasks) == 1:
        axes = [axes]
    
    for i, task in enumerate(tasks):
        task_df = df[df['Task'] == task]
        
        # Pivot for grouped bar chart
        pivot_df = task_df.pivot(index='Model', columns='Metric', values='Score')
        
        # Plot
        pivot_df.plot(kind='bar', ax=axes[i], width=0.8)
        axes[i].set_title(f'{task.capitalize()} Models', fontsize=14)
        axes[i].set_xlabel('Model', fontsize=12)
        axes[i].set_ylabel('Score' if i == 0 else '', fontsize=12)
        axes[i].set_ylim(0, 1.05)
        axes[i].legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved model comparison to {plot_path}")
    return plot_path


def plot_learning_curves(
    model_name: str,
    train_scores: List[float],
    val_scores: List[float],
    train_sizes: List[int],
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Path:
    """
    Plot learning curves showing training progress.
    
    Args:
        model_name: Model name
        train_scores: Training scores at each size
        val_scores: Validation scores at each size
        train_sizes: Training set sizes
        output_dir: Directory to save plot
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Plot curves
    plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training score', linewidth=2, markersize=8)
    plt.plot(train_sizes, val_scores, 'o-', color='green', label='Validation score', linewidth=2, markersize=8)
    
    # Fill between for variance if available
    if isinstance(train_scores[0], (list, np.ndarray)):
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
    
    # Formatting
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Learning Curves - {model_name}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Save plot
    plot_path = output_dir / f"learning_curves_{model_name.replace('/', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved learning curves to {plot_path}")
    return plot_path


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    top_n: int = 20,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Path:
    """
    Plot feature importance for interpretable models.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        model_name: Model name
        top_n: Number of top features to show
        output_dir: Directory to save plot
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get top features
    indices = np.argsort(np.abs(importances))[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Color based on positive/negative
    colors = ['green' if x > 0 else 'red' for x in top_importances]
    
    # Horizontal bar plot
    plt.barh(range(len(top_features)), top_importances, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Features - {model_name}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add zero line
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Invert y-axis to have most important at top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"feature_importance_{model_name.replace('/', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved feature importance to {plot_path}")
    return plot_path


def plot_ensemble_weights(
    model_names: List[str],
    weights: np.ndarray,
    ensemble_name: str,
    output_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Path:
    """
    Plot ensemble model weights.
    
    Args:
        model_names: List of model names
        weights: Model weights
        ensemble_name: Ensemble name
        output_dir: Directory to save plot
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Create bar plot
    x = range(len(model_names))
    plt.bar(x, weights, alpha=0.7, color='steelblue')
    
    # Formatting
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylabel('Weight', fontsize=12)
    plt.title(f'Ensemble Weights - {ensemble_name}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, weight in enumerate(weights):
        plt.text(i, weight + 0.01, f'{weight:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"ensemble_weights_{ensemble_name.replace('/', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Saved ensemble weights to {plot_path}")
    return plot_path


def create_performance_report(
    results: List[Dict[str, Any]],
    output_dir: Optional[Path] = None
) -> Path:
    """
    Create a comprehensive performance report with multiple plots.
    
    Args:
        results: List of all results
        output_dir: Directory to save report
        
    Returns:
        Path to report directory
    """
    if output_dir is None:
        output_dir = CFG.artifacts_dir / "performance_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("📊 Creating performance report...")
    
    # 1. Model comparison
    plot_model_comparison(results, output_dir=output_dir)
    
    # 2. ROC curves for each task
    for task in ['keto', 'vegan']:
        task_results = [r for r in results if r.get('task') == task]
        if task_results:
            plot_roc_curves(task_results, task, output_dir=output_dir)
            plot_precision_recall_curves(task_results, task, output_dir=output_dir)
    
    # 3. Best model confusion matrices
    for task in ['keto', 'vegan']:
        task_results = [r for r in results if r.get('task') == task and 'pred' in r]
        if task_results:
            best_result = max(task_results, key=lambda x: x.get('F1', 0))
            if 'y_true' in best_result:
                plot_confusion_matrix(
                    best_result['y_true'],
                    best_result['pred'],
                    task,
                    best_result['model'],
                    output_dir=output_dir
                )
    
    # 4. Create summary HTML
    create_html_report(results, output_dir)
    
    log.info(f"✅ Performance report saved to {output_dir}")
    return output_dir


def create_html_report(
    results: List[Dict[str, Any]],
    output_dir: Path
):
    """
    Create an HTML summary report.
    
    Args:
        results: List of results
        output_dir: Directory to save report
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diet Classification Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metric { font-weight: bold; }
            img { max-width: 800px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Diet Classification Performance Report</h1>
        <p>Generated: {timestamp}</p>
        
        <h2>Model Performance Summary</h2>
        <table>
            <tr>
                <th>Task</th>
                <th>Model</th>
                <th>F1</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>ROC AUC</th>
            </tr>
    """
    
    # Add results rows
    for result in sorted(results, key=lambda x: (x.get('task', ''), -x.get('F1', 0))):
        html_content += f"""
            <tr>
                <td>{result.get('task', 'N/A')}</td>
                <td>{result.get('model', 'N/A')}</td>
                <td class="metric">{result.get('F1', 0):.3f}</td>
                <td>{result.get('ACC', 0):.3f}</td>
                <td>{result.get('PREC', 0):.3f}</td>
                <td>{result.get('REC', 0):.3f}</td>
                <td>{result.get('ROC', 0):.3f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
    """
    
    # Add images
    for img_file in sorted(output_dir.glob("*.png")):
        html_content += f"""
        <h3>{img_file.stem.replace('_', ' ').title()}</h3>
        <img src="{img_file.name}" alt="{img_file.stem}">
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Format timestamp
    from datetime import datetime
    html_content = html_content.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # Save HTML
    html_path = output_dir / "report.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    log.info(f"Created HTML report: {html_path}")