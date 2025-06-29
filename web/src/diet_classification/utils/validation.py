#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation utilities and pre-flight checks for the pipeline.

Based on original lines 428-631 from diet_classifiers.py
"""

import os
import sys
import platform
import requests
import psutil
import numpy as np
from pathlib import Path

from ..core import log, get_pipeline_state
from ..config import CFG
from .memory import get_available_memory


def preflight_checks():
    """
    Run comprehensive checks before starting training.

    Enhanced with network connectivity checks and better resource validation.

    Returns:
        bool: True if all critical checks pass, False otherwise
        
    Based on original lines 428-631
    """
    issues = []
    warnings = []

    log.info("\n🔍 RUNNING PRE-FLIGHT CHECKS...")

    # Update memory mode
    pipeline_state = get_pipeline_state()
    pipeline_state.update_memory_mode()
    available_gb = get_available_memory()

    # Check memory with configurable thresholds
    if available_gb < 4:
        issues.append(
            f"Insufficient memory: {available_gb:.1f} GB (need at least 4 GB)")
    elif available_gb < CFG.memory_thresholds['medium']:
        warnings.append(
            f"Low memory: {available_gb:.1f} GB (recommend {CFG.memory_thresholds['medium']}+ GB)")
    else:
        log.info(
            f"   ✅ Memory: {available_gb:.1f} GB available ({pipeline_state.memory_mode} mode)")

    # Check disk space
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    if free_gb < 5:
        issues.append(
            f"Insufficient disk space: {free_gb:.1f} GB free (need at least 5 GB)")
    else:
        log.info(f"   ✅ Disk space: {free_gb:.1f} GB free")

    # Check temporary directory space
    try:
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_usage = psutil.disk_usage(temp_dir)
        temp_free_gb = temp_usage.free / (1024**3)
        if temp_free_gb < 2:
            warnings.append(
                f"Low temp space: {temp_free_gb:.1f} GB free in {temp_dir}")
        else:
            log.info(f"   ✅ Temp directory: {temp_free_gb:.1f} GB free")
    except Exception as e:
        warnings.append(f"Could not check temp directory: {e}")

    # Check data files
    required_files = [
        CFG.url_map["allrecipes.parquet"],
        CFG.url_map["ground_truth_sample.csv"]
    ]

    for file_path in required_files:
        if not file_path.startswith('http') and not Path(file_path).exists():
            issues.append(f"Missing required file: {file_path}")
        else:
            log.info(f"   ✅ Data file found: {Path(file_path).name}")

    # Check USDA data
    usda_files = ["food.csv", "food_nutrient.csv", "nutrient.csv"]
    usda_missing = [f for f in usda_files if not (CFG.usda_dir / f).exists()]
    if usda_missing:
        warnings.append(
            f"Missing USDA files: {usda_missing} (will skip carb-based rules)")
    else:
        log.info(f"   ✅ USDA nutritional data: All files present")

    # Check network connectivity
    log.info(f"   🌐 Checking network connectivity...")
    network_ok = False
    for url in CFG.network_config['connectivity_test_urls']:
        try:
            response = requests.head(url, timeout=5)
            if response.status_code < 500:
                network_ok = True
                log.info(f"   ✅ Network connectivity: OK (tested {url})")
                break
        except Exception:
            continue

    if not network_ok:
        warnings.append(
            "No network connectivity detected - image downloads will fail")
        log.warning(f"   ⚠️  Network connectivity check failed")

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            log.info(f"   ✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            warnings.append("No GPU available - image processing will be slower")
    except ImportError:
        warnings.append("PyTorch not installed - GPU features disabled")

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        issues.append(
            f"Python {python_version.major}.{python_version.minor} detected - need Python 3.8+")
    else:
        log.info(
            f"   ✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check critical dependencies
    try:
        import sklearn
        sklearn_version = sklearn.__version__
        log.info(f"   ✅ scikit-learn: {sklearn_version}")
    except ImportError:
        issues.append("scikit-learn not installed - ML features disabled")

    # Check artifacts directory permissions
    try:
        test_file = CFG.artifacts_dir / ".write_test"
        CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test")
        test_file.unlink()
        log.info(f"   ✅ Artifacts directory writable: {CFG.artifacts_dir}")
    except Exception as e:
        issues.append(f"Cannot write to artifacts directory: {e}")

    # Check Docker volume mounts
    import os
    if os.path.exists('/.dockerenv'):
        log.info(f"   🐳 Running in Docker container")
        # Check if volumes are properly mounted
        volume_checks = [
            (CFG.data_dir, "data"),
            (CFG.artifacts_dir, "artifacts"),
            (CFG.state_dir, "pipeline_state")
        ]
        for vol_path, vol_name in volume_checks:
            if not vol_path.exists():
                warnings.append(
                    f"Docker volume {vol_name} may not be mounted correctly")
            else:
                log.info(f"   ✅ Docker volume {vol_name}: mounted")

    # Check for existing models
    models_exist = (CFG.artifacts_dir / "models.pkl").exists() and (
        CFG.artifacts_dir / "vectorizer.pkl").exists()
    pretrained_exist = (CFG.pretrained_models_dir / "models.pkl").exists() and (
        CFG.pretrained_models_dir / "vectorizer.pkl").exists()

    if models_exist:
        log.info(f"   ✅ Trained models found in artifacts/")
    elif pretrained_exist:
        log.info(f"   ✅ Pretrained models found in pretrained_models/")
    else:
        warnings.append(
            "No existing models found - will need to train from scratch")

    # Report results
    if issues:
        log.error("\n❌ PRE-FLIGHT CHECK FAILED:")
        for issue in issues:
            log.error(f"   ├─ {issue}")
        log.error(f"   └─ Please fix these issues before proceeding")
        return False

    log.info("\n✅ PRE-FLIGHT CHECK PASSED")

    if warnings:
        log.warning("\n⚠️  Warnings (non-critical):")
        for warning in warnings:
            log.warning(f"   ├─ {warning}")
        log.warning(
            f"   └─ Pipeline will continue but may have reduced functionality")

    # System summary
    log.info("\n📊 SYSTEM SUMMARY:")
    log.info(f"   ├─ CPU cores: {psutil.cpu_count()}")
    log.info(
        f"   ├─ Total memory: {psutil.virtual_memory().total // (1024**3)} GB")
    log.info(f"   ├─ Available memory: {available_gb:.1f} GB")
    log.info(f"   ├─ Memory mode: {pipeline_state.memory_mode}")
    log.info(f"   ├─ Platform: {platform.system()} {platform.release()}")
    log.info(
        f"   └─ Docker: {'Yes' if os.path.exists('/.dockerenv') else 'No'}")

    return True


def tune_threshold(y_true, probs):
    """
    Find optimal classification threshold using precision-recall curve.

    Instead of using 0.5, this finds the threshold that maximizes F1 score.

    Args:
        y_true: True labels
        probs: Predicted probabilities

    Returns:
        Optimal threshold value
        
    Based on original lines 5608-5628
    """
    try:
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        log.warning("scikit-learn not available, using default threshold 0.5")
        return 0.5
    
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5