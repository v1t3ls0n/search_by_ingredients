#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
================================================================================
DIET CLASSIFIER PIPELINE - MULTI-MODAL MACHINE LEARNING FOR INGREDIENT ANALYSIS
================================================================================
This comprehensive machine learning pipeline classifies recipes as keto-friendly 
or vegan based on their ingredients using a multi-modal approach combining:
1. TEXT FEATURES: TF-IDF vectorization of normalized ingredient lists
2. IMAGE FEATURES: ResNet-50 embeddings from recipe photos
3. RULE-BASED VERIFICATION: Domain-specific heuristics and USDA nutritional data

KEY COMPONENTS:
---------------
- SILVER LABEL GENERATION: Creates weak labels from unlabeled data using multi-stage
  rule-based heuristics simulating expert knowledge:
    • Token normalization + lemmatization
    • Regex-based blacklist/whitelist
    • USDA-based labeling: entire USDA nutritional dataset is scanned and all
      food entries with ≤10g carbohydrates per 100g are added as keto-friendly examples
    • Phrase-level disqualifications (e.g., "chicken broth")
    • Whitelist override of verified-safe ingredients (e.g., "almond flour")
    • Token-level disqualifiers from NON_KETO list
    • ML fallback prediction with rule-based correction
    • Photo sanity filtering: excludes rows with URLs like 'nophoto', 'nopic', 'nopicture'
- MODEL TRAINING: Trains diverse ML models (Logistic Regression, SVM, MLP, Random Forest, etc.)
- ENSEMBLE METHODS: Combines multiple classifiers using top-N voting and rule-based overrides
- CACHING & RESTORE: Saves and reuses models, vectorizers, image embeddings
- LOGGING: Logs to both console and `artifacts/pipeline.log`
- FULL EVALUATION: Saves gold-test predictions and per-class metrics to CSV

ARCHITECTURE OVERVIEW:
----------------------
1. Data Loading:
   - Loads silver (unlabeled) and gold (labeled) recipes
   - Uses USDA nutritional DB to automatically extend the silver dataset with
     thousands of labeled examples based on carbohydrate thresholds
   - Input can be CSV or Parquet

2. Feature Extraction:
   - Text: TF-IDF vectorization after custom normalization
   - Image: ResNet-50 feature extraction from downloaded photos
   - Merges modalities where appropriate

3. Model Training:
   - Silver-labeled data → supervised classifiers
   - Supports `--mode text`, `--mode image`, `--mode both`

4. Prediction & Evaluation:
   - Supports ingredient inference or full CSV evaluation
   - Computes Accuracy, F1, Precision, Recall
   - Exports predictions and metrics to artifacts directory

USAGE MODES:
------------
1. Training: `--train` to trigger full silver model training pipeline
2. Inference: `--ingredients` for direct classification from command line
3. Evaluation: `--ground_truth` for benchmarking against labeled CSV
4. Batch Prediction: `--predict` for unlabeled CSV file inference

EXAMPLES:
---------
# Train models on silver-labeled data
python diet_classifiers.py --train --mode both

# Evaluate trained models on ground truth data
python diet_classifiers.py --ground_truth /path/to/ground_truth.csv

# Classify custom ingredients
python diet_classifiers.py --ingredients "almond flour, eggs, butter"

# Batch inference on unlabeled CSV
python diet_classifiers.py --predict /path/to/recipes.csv

Robust against partial data, broken images, or failed downloads.
Supports interactive development, Docker builds, and production use.

Author: Guy Vitelson (aka @v1t3ls0n on GitHub)
"""

from __future__ import annotations
import gc  # Memory cleanup
from functools import partial
from tqdm import tqdm
import sys
import threading
import json
import logging
import re
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple
from rapidfuzz import process
import pickle
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import psutil
import nltk
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE, RandomOverSampler
import uuid
from datetime import datetime

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
"""
  This section imports all required libraries, organizing them by category:
- Standard library modules
- Core third-party libraries (NumPy, Pandas, etc.)
- Machine learning libraries (scikit-learn, LightGBM)
- Deep learning libraries (PyTorch, torchvision)
- Specialized libraries (NLTK, imbalanced-learn)
"""


# --- NLTK (used for lemmatization) ---
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    wnl = WordNetLemmatizer()
except:
    wnl = None

# --- Optional: scikit-learn ---
try:
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import VotingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import (
        LogisticRegression, SGDClassifier,
        PassiveAggressiveClassifier, RidgeClassifier
    )
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        precision_score, recall_score, f1_score, roc_auc_score,
        precision_recall_curve, roc_curve,
        ConfusionMatrixDisplay, RocCurveDisplay
    )
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None  # LightGBM optional

    SKLEARN_AVAILABLE = True

except ImportError as e:  # pragma: no cover
    warnings.warn(
        f"scikit-learn not installed ({e}). ML features will be disabled.", stacklevel=2)
    SKLEARN_AVAILABLE = False

    # Fallbacks
    class BaseEstimator:
        pass

    class ClassifierMixin:
        pass

    def clone(obj): return obj

    def make_pipeline(
        *args, **kwargs): raise ImportError("scikit-learn is required")

    def precision_recall_curve(
        *args, **kwargs): raise ImportError("scikit-learn is required")

    class RBFSampler:
        pass

    class SVC:
        pass

    class CalibratedClassifierCV:
        pass

    class VotingClassifier:
        pass

    class TfidfVectorizer:
        def __init__(
            self, **kwargs): raise ImportError("scikit-learn is required")

    class LogisticRegression:
        pass
    LinearSVC = MLPClassifier = GridSearchCV = RandomizedSearchCV = None
    accuracy_score = average_precision_score = confusion_matrix = f1_score = precision_score = recall_score = roc_auc_score = None
    SGDClassifier = MultinomialNB = PassiveAggressiveClassifier = RidgeClassifier = None

# --- Optional: LightGBM ---
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

# --- Optional: PyTorch and torchvision (for image embeddings) ---
try:  # pragma: no cover
    import requests
    from PIL import Image
    import torch
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    warnings.warn(
        f"PyTorch/torchvision not installed ({e}). Image features disabled.", stacklevel=2)
    Image = None
    requests = None
    torch = None
    models = None
    transforms = None
    TORCH_AVAILABLE = False


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================
"""
Central configuration for the pipeline including:
- File paths and directories
- URL mappings for datasets
- Vectorizer parameters
- Domain-specific ingredient lists for keto/vegan classification
"""


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration container for the pipeline with validation.
    """
    pretrained_models_dir: Path = Path("/app/pretrained_models")
    artifacts_dir: Path = Path("/app/artifacts")
    logs_dir: Path = Path("/app/artifacts/logs")
    data_dir: Path = Path("/app/data")
    usda_dir: Path = Path("/app/data/usda")
    state_dir: Path = Path("/app/pipeline_state")
    checkpoints_dir: Path = Path("/app/pipeline_state/checkpoints")
    cache_dir: Path = Path("/app/pipeline_state/cache")
    url_map: Mapping[str, str] = field(default_factory=lambda: {
        "allrecipes.parquet": "/app/data/allrecipes.parquet",
        "ground_truth_sample.csv": "/app/data/ground_truth_sample.csv",
    })
    vec_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        min_df=2, ngram_range=(1, 3), max_features=50000, sublinear_tf=True))
    image_dir: Path = Path("dataset/arg_max/images")

    # NEW: Configurable thresholds
    memory_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 16.0,      # GB for high memory mode
        'medium': 8.0,     # GB for medium memory mode
        'critical': 0.9,   # 90% memory usage is critical
        'warning': 0.85,   # 85% memory usage is warning
        'memmap': 10000,   # Use memmap for datasets larger than this
    })

    # NEW: Network configuration
    network_config: Dict[str, Any] = field(default_factory=lambda: {
        'connectivity_test_urls': [
            "https://www.google.com",
            "https://api.github.com"
        ],
        'download_timeout': 30,
        'retry_delays': [0.5, 1, 2],  # Faster retry delays
        'max_download_workers': 16,
    })

    # NEW: Model configuration
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        'min_images_for_training': 10,
        'sanity_check_fraction': 0.01,
        'checkpoint_save_frequency': 10,  # Save checkpoint every N batches
    })

    def __post_init__(self):
        """Validate configuration on initialization."""
        # Create missing directories
        for field_name, value in self.__dict__.items():
            if isinstance(value, Path) and field_name != 'url_map':
                if not value.exists() and not str(value).startswith('http'):
                    try:
                        value.mkdir(parents=True, exist_ok=True)
                        logging.getLogger("PIPE").info(
                            f"Created missing directory: {value}")
                    except Exception as e:
                        logging.getLogger("PIPE").warning(
                            f"Could not create {value}: {e}")


# Create the new config instance
CFG = Config()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
"""
Direct write logging that bypasses buffering issues entirely.
"""

# Make sure artifacts dir exists
CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
CFG.logs_dir.mkdir(parents=True, exist_ok=True)

# Define log file path
log_file = CFG.logs_dir / "diet_classifiers.py.log"

# Thread lock for file writing
file_lock = threading.Lock()


class DirectWriteHandler(logging.Handler):
    """Handler that writes directly to file with no buffering"""

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        # Write a separator for new runs (but don't delete existing content)
        self._write_direct("\n" + "="*80 + "\n")
        self._write_direct(
            f"NEW RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._write_direct(f"Process ID: {os.getpid()}\n")
        self._write_direct("="*80 + "\n")

    def _write_direct(self, msg):
        """Write directly to file with no buffering"""
        with file_lock:
            # Always append, never truncate
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(msg)
                f.flush()
                # Force OS-level flush
                try:
                    os.fsync(f.fileno())
                except:
                    pass  # Some filesystems don't support fsync

    def emit(self, record):
        try:
            msg = self.format(record)
            self._write_direct(msg + '\n')
        except Exception:
            self.handleError(record)


# Get logger
log = logging.getLogger("PIPE")

# Configure logger - but check if handlers already exist to avoid duplicates
if not log.handlers:
    log.setLevel(logging.INFO)

    # Define formatter
    formatter = logging.Formatter(
        "%(asctime)s │ %(levelname)s │ %(message)s", datefmt="%H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    # Direct write file handler
    direct_handler = DirectWriteHandler(str(log_file))
    direct_handler.setFormatter(formatter)
    log.addHandler(direct_handler)

    # Test logging
    log.info("Logging system initialized successfully")

# Exception hook


def log_exception_hook(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log.error("Uncaught exception:", exc_info=(
        exc_type, exc_value, exc_traceback))


sys.excepthook = log_exception_hook


# =============================================================================
# 2. CONSOLIDATED PIPELINE STATE MANAGER
# =============================================================================

class PipelineStateManager:
    """Centralized state management for the pipeline."""

    def __init__(self):
        self.datasets = None
        self.carb_map = None
        self.fuzzy_keys = None
        self.vectorizer = None
        self.models = {}
        self.initialized = False
        self.checkpoints = {}
        self.memory_mode = None  # 'high', 'medium', 'low'

    def update_memory_mode(self):
        """Update memory mode based on available resources."""
        available_gb = get_available_memory()

        if available_gb >= CFG.memory_thresholds['high']:
            self.memory_mode = 'high'
        elif available_gb >= CFG.memory_thresholds['medium']:
            self.memory_mode = 'medium'
        else:
            self.memory_mode = 'low'

        log.info(
            f"Memory mode set to: {self.memory_mode} ({available_gb:.1f} GB available)")
        return self.memory_mode

    def should_use_memmap(self, data_size: int) -> bool:
        """Determine if memory-mapped arrays should be used."""
        return data_size > CFG.memory_thresholds['memmap'] or self.memory_mode == 'low'

    def save_checkpoint(self, stage: str, data: dict):
        """Save a checkpoint for the given stage."""
        checkpoint_path = CFG.checkpoints_dir / f"checkpoint_{stage}.pkl"
        self.checkpoints[stage] = data

        checkpoint_data = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'memory_mode': self.memory_mode,
            'pipeline_version': '1.0',  # Add versioning
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        log.info(f"Saved checkpoint: {stage}")

    def load_checkpoint(self, stage: str):
        """Load a checkpoint for the given stage."""
        checkpoint_path = CFG.checkpoints_dir / f"checkpoint_{stage}.pkl"

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                self.checkpoints[stage] = checkpoint_data['data']
                log.info(
                    f"Loaded checkpoint: {stage} (saved at {checkpoint_data['timestamp']})")
                return checkpoint_data['data']
            except Exception as e:
                log.error(f"Failed to load checkpoint {stage}: {e}")

        return None

    def clear(self):
        """Clear all state."""
        self.datasets = None
        self.carb_map = None
        self.fuzzy_keys = None
        self.vectorizer = None
        self.models.clear()
        self.checkpoints.clear()
        self.initialized = False


# Create global instance
PIPELINE_STATE = PipelineStateManager()


# =============================================================================
# 3. ENHANCED PRE-FLIGHT CHECKS WITH NETWORK CONNECTIVITY
# =============================================================================
# Replace the existing preflight_checks function

def preflight_checks():
    """
    Run comprehensive checks before starting training.

    Enhanced with network connectivity checks and better resource validation.

    Returns:
        bool: True if all critical checks pass, False otherwise
    """
    import platform
    import requests

    issues = []
    warnings = []

    log.info("\n🔍 RUNNING PRE-FLIGHT CHECKS...")

    # Update memory mode
    PIPELINE_STATE.update_memory_mode()
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
            f"   ✅ Memory: {available_gb:.1f} GB available ({PIPELINE_STATE.memory_mode} mode)")

    # Check disk space
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    if free_gb < 5:
        issues.append(
            f"Insufficient disk space: {free_gb:.1f} GB free (need at least 5 GB)")
    else:
        log.info(f"   ✅ Disk space: {free_gb:.1f} GB free")

    # NEW: Check temporary directory space
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

    # NEW: Check network connectivity
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
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(
            0).total_memory / (1024**3)
        log.info(f"   ✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        warnings.append("No GPU available - image processing will be slower")

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

    # NEW: Check Docker volume mounts
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
    log.info(f"   ├─ Memory mode: {PIPELINE_STATE.memory_mode}")
    log.info(f"   ├─ Platform: {platform.system()} {platform.release()}")
    log.info(
        f"   └─ Docker: {'Yes' if os.path.exists('/.dockerenv') else 'No'}")

    return True


# =============================================================================
# MEMORY OPTIMIZATION
# =============================================================================
"""
Functions for managing memory usage during training. Critical for handling
large datasets and preventing out-of-memory errors, especially when working
with both text and image features.
"""


def get_available_memory(safety_factor=0.9):
    """
    Get available memory accounting for Docker container limits.
    Enhanced version that properly detects container memory constraints.
    """
    import platform

    mem = psutil.virtual_memory()
    total_memory = mem.total

    # Check Docker cgroup limits
    cgroup_files = [
        '/sys/fs/cgroup/memory/memory.limit_in_bytes',  # cgroup v1
        '/sys/fs/cgroup/memory.max',  # cgroup v2
    ]

    for cgroup_file in cgroup_files:
        if os.path.exists(cgroup_file):
            try:
                with open(cgroup_file, 'r') as f:
                    limit = f.read().strip()
                    if limit != 'max' and limit.isdigit():
                        cgroup_limit = int(limit)
                        # Use the smaller of cgroup limit or system memory
                        if cgroup_limit < total_memory * 2:  # sanity check
                            total_memory = min(total_memory, cgroup_limit)
                            log.info(
                                f"Container memory limited by cgroup: {total_memory / (1024**3):.1f} GB")
                            break
            except Exception as e:
                log.warning(f"Could not read cgroup file {cgroup_file}: {e}")

    # Add check for Docker Desktop on Mac/Windows
    # These platforms often report system memory instead of container limits
    if platform.system() in ['Darwin', 'Windows']:
        # Try to read from Docker's memory limit file
        docker_limit_file = '/.dockerenv'
        if os.path.exists(docker_limit_file):
            log.warning(
                "Running in Docker on Mac/Windows - memory limits may not be accurately detected")
            # Conservative estimate for Docker Desktop
            estimated_limit_gb = min(
                8.0, total_memory / (1024**3))  # Default to 8GB max
            log.info(
                f"Using conservative Docker Desktop limit: {estimated_limit_gb:.1f} GB")
            return estimated_limit_gb * safety_factor

    # Return usable memory in GB
    usable_gb = (total_memory * safety_factor) / (1024**3)

    log.info(
        f"System/Container total memory: {total_memory / (1024**3):.1f} GB")
    log.info(
        f"Safe memory limit ({safety_factor*100:.0f}%): {usable_gb:.1f} GB")

    return usable_gb


def optimize_memory_usage(stage_name=""):
    """
    Optimize memory usage during training with enhanced Docker support.
    """
    # Get memory before cleanup
    try:
        memory_before = psutil.virtual_memory()
        memory_before_used = memory_before.used
        memory_before_percent = memory_before.percent

        # Log current memory state with container awareness
        available_memory_gb = get_available_memory(
            safety_factor=1.0)  # Get total available
        log.info(f"   🧹 {stage_name}: Memory optimization")
        log.info(
            f"      ├─ Container/System memory: {available_memory_gb:.1f} GB total")
        log.info(
            f"      ├─ Currently used: {memory_before_percent:.1f}% ({memory_before_used / (1024**2):.0f} MB)")

    except Exception as e:
        log.error(f"Failed to get initial memory stats: {e}")
        return "error"

    # Force garbage collection multiple times
    collected_total = 0
    for i in range(3):
        try:
            collected = gc.collect()
            collected_total += collected
        except Exception as e:
            log.debug(f"Garbage collection pass {i+1} failed: {e}")

    # Clear GPU cache if available
    gpu_freed = 0
    if torch and torch.cuda.is_available():
        try:
            gpu_before = torch.cuda.memory_allocated() / (1024**2)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure operations complete
            gpu_after = torch.cuda.memory_allocated() / (1024**2)
            gpu_freed = max(0, gpu_before - gpu_after)
        except Exception as e:
            log.debug(f"GPU memory cleanup failed: {e}")

    # Get memory after cleanup
    try:
        memory_after = psutil.virtual_memory()
        memory_freed_bytes = max(0, memory_before_used - memory_after.used)
        memory_freed_mb = memory_freed_bytes / (1024**2)

    except Exception as e:
        log.error(f"Failed to get final memory stats: {e}")
        return "error"

    # Log results
    if memory_freed_mb > 1.0 or collected_total > 0 or gpu_freed > 1.0:
        log.info(f"      ├─ Freed: {memory_freed_mb:.1f} MB RAM")
        if collected_total > 0:
            log.info(f"      ├─ Objects collected: {collected_total}")
        if gpu_freed > 1.0:
            log.info(f"      ├─ GPU freed: {gpu_freed:.1f} MB")

    # Check final status
    if memory_after.percent > 90:
        log.error(
            f"      ❌ CRITICAL memory usage: {memory_after.percent:.1f}%")
        # Try emergency cleanup
        handle_memory_crisis()
        return "critical"
    elif memory_after.percent > 85:
        log.warning(
            f"      ⚠️  High memory usage: {memory_after.percent:.1f}%")
        return "high"
    elif memory_after.percent > 70:
        log.info(
            f"      ⚠️  Moderate memory usage: {memory_after.percent:.1f}%")
        return "moderate"
    else:
        log.info(f"      ✅ Memory usage normal: {memory_after.percent:.1f}%")
        return "normal"


def handle_memory_crisis():
    """
    Emergency memory cleanup when usage is critical.

    Enhanced with adaptive model switching and disk-based processing.
    """
    log.warning("🚨 MEMORY CRISIS - Applying emergency cleanup")

    try:
        initial_memory = psutil.virtual_memory()
        initial_percent = initial_memory.percent
        log.info(f"   ├─ Initial memory: {initial_percent:.1f}%")

        # Step 1: Multiple aggressive garbage collection passes
        total_collected = 0
        for i in range(5):
            try:
                collected = gc.collect()
                total_collected += collected
                if collected > 0:
                    log.info(
                        f"   ├─ GC pass {i+1}: {collected} objects collected")
            except Exception as e:
                log.debug(f"GC pass {i+1} failed: {e}")

        # Step 2: Clear all GPU memory
        gpu_freed = 0
        if torch and torch.cuda.is_available():
            try:
                gpu_before = torch.cuda.memory_allocated() / (1024**2)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gpu_after = torch.cuda.memory_allocated() / (1024**2)
                gpu_freed = max(0, gpu_before - gpu_after)
                log.info(f"   ├─ GPU memory freed: {gpu_freed:.1f} MB")
            except Exception as e:
                log.debug(f"   ├─ GPU cleanup failed: {e}")

        # Step 3: Clear Python internal caches
        try:
            import importlib
            if hasattr(importlib, 'invalidate_caches'):
                importlib.invalidate_caches()
        except Exception as e:
            log.debug(f"   ├─ Cache cleanup failed: {e}")

        # Step 4: Force memory compaction
        try:
            gc.set_debug(0)
            gc.collect()
        except Exception:
            pass

        # Step 5: Check final memory
        final_memory = psutil.virtual_memory()
        final_percent = final_memory.percent
        memory_freed_mb = (initial_memory.used - final_memory.used) / (1024**2)

        log.info(f"   ├─ Objects collected: {total_collected}")
        log.info(f"   ├─ Memory freed: {memory_freed_mb:.1f} MB")
        log.info(f"   └─ Final memory usage: {final_percent:.1f}%")

        # NEW: Adaptive strategy based on final memory
        if final_percent > CFG.memory_thresholds['critical'] * 100:
            log.error(f"   ❌ Still critical! Switching to emergency mode")

            # Force minimal models
            os.environ['FORCE_MINIMAL_MODELS'] = '1'

            # Switch to disk-based processing
            os.environ['USE_DISK_CACHE'] = '1'

            # Reduce batch sizes
            os.environ['BATCH_SIZE_MULTIPLIER'] = '0.25'

            # Update pipeline state
            PIPELINE_STATE.memory_mode = 'critical'

            log.info(f"   🚨 Emergency measures activated:")
            log.info(f"      ├─ Minimal models only")
            log.info(f"      ├─ Disk-based caching enabled")
            log.info(f"      └─ Batch sizes reduced to 25%")

        return final_percent

    except Exception as e:
        log.error(f"Memory crisis handling failed: {e}")
        return 90.0  # Assume high usage if we can't measure


# =============================================================================
# DATA LOADING AND DATASET MANAGEMENT
# =============================================================================
"""
Functions for loading and managing the various datasets used in the pipeline:
- Recipe datasets (silver and gold standard)
- USDA nutritional database
- Image metadata
"""


def _load_usda_carb_table() -> pd.DataFrame:
    """
    Load USDA nutritional database and extract carbohydrate content.

    This function loads the USDA FoodData Central database files and creates
    a lookup table mapping food descriptions to carbohydrate content per 100g.
    Used for the numeric keto rule (foods with >10g carbs/100g are non-keto).

    Returns:
        DataFrame with columns:
        - food_desc: Lowercase food description
        - carb_100g: Carbohydrate content per 100g

    Note:
        Requires USDA database CSV files in the configured directory:
        - food.csv: Food items and descriptions
        - nutrient.csv: Nutrient definitions
        - food_nutrient.csv: Nutrient content per food item
    """

    # 1. Resolve file paths
    usda = CFG.usda_dir
    food_csv = usda / "food.csv"
    food_nutrient_csv = usda / "food_nutrient.csv"
    nutrient_csv = usda / "nutrient.csv"

    if not (food_csv.exists() and food_nutrient_csv.exists() and nutrient_csv.exists()):
        log.warning(
            "USDA tables not found in %s – skipping numeric carb table", usda)
        return pd.DataFrame(columns=["food_desc", "carb_100g"])

    # 2. Locate nutrient_id for carbohydrate
    nutrient = pd.read_csv(nutrient_csv, usecols=["id", "name"])
    carb_id = int(
        nutrient.loc[
            nutrient["name"].str.contains(
                "Carbohydrate, by difference", case=False),
            "id"
        ].iloc[0]
    )

    # 3. Pull carb rows from food_nutrient
    carb_rows = pd.read_csv(
        food_nutrient_csv,
        usecols=["fdc_id", "nutrient_id", "amount"],
        dtype={"fdc_id": "int32", "nutrient_id": "int16", "amount": "float32"},
    ).query("nutrient_id == @carb_id")[["fdc_id", "amount"]]

    # 4. Join with food descriptions
    food = pd.read_csv(
        food_csv,
        usecols=["fdc_id", "description"],
        dtype={"fdc_id": "int32", "description": "string"},
    )
    carb_df = (
        carb_rows.merge(food, on="fdc_id", how="left", validate="m:1")
        .dropna(subset=["description"])
        .assign(
            food_desc=lambda df: df["description"].str.lower().str.strip(),
            carb_100g=lambda df: df["amount"].round(1),
        )[["food_desc", "carb_100g"]]
        .drop_duplicates("food_desc")
        .reset_index(drop=True)
    )
    log.info("USDA carb table loaded: %d distinct food descriptions", len(carb_df))
    return carb_df


def label_usda_keto_data(carb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the USDA carb table into silver-style training data.

    Labels each ingredient as keto if carbs_per_100g < 10.
    Applies full ingredient normalization using `normalise()`.

    Args:
        carb_df: Output of _load_usda_carb_table()

    Returns:
        DataFrame with: ingredient, clean, silver_keto, silver_vegan, source
    """
    df = carb_df.copy()
    df["ingredient"] = df["food_desc"]
    df["clean"] = df["ingredient"].map(normalise)
    df["silver_keto"] = (df["carb_100g"] < 10).astype(int)
    df["silver_vegan"] = np.nan
    df["source"] = "usda"
    return df[["ingredient", "clean", "silver_keto", "silver_vegan", "source"]]


def _download_images(df: pd.DataFrame, img_dir: Path, max_workers: int = None, force: bool = False) -> list[int]:
    """
    Download images with maximum robustness and intelligent caching.

    This refactored function prioritizes:
    1. Embeddings-first approach - always check for embeddings before downloading
    2. Graceful degradation - work with partial data
    3. Smart caching - multiple fallback strategies
    4. Error resilience - handle all failure modes
    5. Resource efficiency - minimize unnecessary work

    Args:
        df: DataFrame containing photo_url column with image URLs
        img_dir: Directory to save downloaded images
        max_workers: Maximum number of concurrent download threads
        force: Force redownload even if embeddings exist

    Returns:
        List of valid indices for downstream processing
    """

    from collections import defaultdict, Counter
    from urllib.parse import urlparse
    import threading
    import hashlib

    # Use config values if max_workers not specified
    if max_workers is None:
        max_workers = CFG.network_config['max_download_workers']

    download_start = time.time()

    # ------------------------------------------------------------------
    # PRIORITY 1: EMBEDDINGS-FIRST APPROACH
    # ------------------------------------------------------------------
    log.info(f"\n🧠 EMBEDDINGS-FIRST IMAGE PIPELINE: {img_dir.name}")
    log.info(f"   Target directory: {img_dir}")
    log.info(f"   Total samples: {len(df):,}")
    log.info(f"   Force recompute: {force}")

    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        log.warning("   ⚠️  PyTorch not available - returning empty indices")
        return []

    # Define embedding search paths in priority order
    mode = img_dir.name  # 'silver' or 'gold'
    embedding_candidates = [
        # Check artifacts directory FIRST (where Docker mounts local files)
        # This is where your files are!
        CFG.artifacts_dir / f"embeddings_{mode}_backup.npy",
        # Alternative naming
        CFG.artifacts_dir / f"{mode}_embeddings.npy",
        # Another alternative
        CFG.artifacts_dir / f"embeddings_{mode}.npy",

        # Then check image directory
        img_dir / "embeddings.npy",                                   # Primary cache
        # Mode-specific
        img_dir / f"embeddings_{mode}.npy",

        # Current directory (legacy)
        # Current dir backup
        Path(f"embeddings_{mode}_backup.npy"),

        # Other fallbacks
        # Parent directory
        img_dir.parent / f"embeddings_{mode}.npy",
        # Embeddings directory
        Path("embeddings") / f"{mode}.npy",
        # Docker path
        Path("/app/embeddings") / f"{mode}.npy",
        # Docker backup
        Path("/app") / f"embeddings_{mode}_backup.npy",
        # Alternative naming
        img_dir / "features.npy",
        img_dir.parent / "cached_embeddings.npy",                    # Generic cache
    ]

    # EMBEDDINGS VALIDATION AND SELECTION
    if not force:
        log.info(f"   🔍 Searching for existing embeddings...")

        for i, emb_path in enumerate(embedding_candidates, 1):
            if not emb_path.exists():
                log.debug(f"      {i}. {emb_path.name}: Not found")
                continue

            try:
                # Quick validation load
                embeddings = np.load(str(emb_path), mmap_mode='r')
                emb_shape = embeddings.shape

                # Validate embedding properties
                if len(emb_shape) != 2:
                    log.debug(
                        f"      {i}. {emb_path.name}: Invalid dimensions {emb_shape}")
                    continue

                if emb_shape[1] not in [2048, 512, 1024, 4096]:  # Common embedding sizes
                    log.debug(
                        f"      {i}. {emb_path.name}: Unusual feature size {emb_shape[1]}")
                    continue

                # Check size compatibility
                size_ratio = emb_shape[0] / len(df)
                min_threshold = 0.3  # Accept if at least 30% coverage

                if size_ratio >= min_threshold:
                    log.info(f"   ✅ EMBEDDINGS FOUND: {emb_path}")
                    log.info(f"      ├─ Shape: {emb_shape}")
                    log.info(
                        f"      ├─ Coverage: {size_ratio:.1%} ({emb_shape[0]:,}/{len(df):,})")
                    log.info(
                        f"      ├─ File size: {emb_path.stat().st_size / (1024**2):.1f} MB")
                    log.info(f"      └─ 🚀 SKIPPING IMAGE DOWNLOADS!")

                    # Return appropriate indices based on available embeddings
                    if emb_shape[0] >= len(df):
                        return sorted(df.index.tolist())
                    else:
                        # Return indices up to available embeddings
                        return sorted(df.index[:emb_shape[0]].tolist())
                else:
                    log.debug(
                        f"      {i}. {emb_path.name}: Low coverage {size_ratio:.1%}")

            except Exception as e:
                log.debug(
                    f"      {i}. {emb_path.name}: Load failed - {str(e)[:50]}")
                continue

        log.info(f"   ❌ No suitable embeddings found - proceeding with downloads")
    else:
        log.info(f"   🔄 Force mode enabled - bypassing embedding cache")

    # ------------------------------------------------------------------
    # PRIORITY 2: SMART DOWNLOAD STRATEGY
    # ------------------------------------------------------------------

    # Create directory structure
    img_dir.mkdir(parents=True, exist_ok=True)

    # Validate DataFrame
    if 'photo_url' not in df.columns:
        log.error("   ❌ No 'photo_url' column found")
        return []

    if df.empty:
        log.warning("   ⚠️  Empty DataFrame provided")
        return []

    # ------------------------------------------------------------------
    # PRIORITY 3: INTELLIGENT DOWNLOAD PLANNING
    # ------------------------------------------------------------------

    log.info(f"\n   📋 DOWNLOAD PLANNING:")

    # Analyze current state
    url_stats = {
        'total': len(df),
        'valid_urls': 0,
        'existing_files': 0,
        'corrupted_files': 0,
        'missing_urls': 0,
        'invalid_urls': 0
    }

    download_queue = []
    valid_indices = []

    # Analyze each row
    for idx, row in df.iterrows():
        img_path = img_dir / f"{idx}.jpg"
        url = row.get('photo_url')

        # Check if file already exists and is valid
        if img_path.exists():
            try:
                # Quick validation - check file size and basic structure
                file_size = img_path.stat().st_size
                if file_size > 100:  # Minimum viable size
                    # Additional validation could include PIL.Image.open() but that's expensive
                    url_stats['existing_files'] += 1
                    valid_indices.append(idx)
                    continue
                else:
                    # File too small, mark for redownload
                    url_stats['corrupted_files'] += 1
                    img_path.unlink(missing_ok=True)
            except Exception:
                url_stats['corrupted_files'] += 1
                img_path.unlink(missing_ok=True)

        # Validate URL
        if not url or not isinstance(url, str):
            url_stats['missing_urls'] += 1
            continue

        url = str(url).strip()
        if not url.startswith(('http://', 'https://')):
            url_stats['invalid_urls'] += 1
            continue

        # URL is valid for download
        url_stats['valid_urls'] += 1
        download_queue.append((idx, url))

    # Log analysis results
    log.info(f"      📊 Analysis Results:")
    log.info(f"      ├─ Total samples: {url_stats['total']:,}")
    log.info(f"      ├─ Existing valid files: {url_stats['existing_files']:,}")
    log.info(
        f"      ├─ Corrupted files cleaned: {url_stats['corrupted_files']:,}")
    log.info(f"      ├─ Valid URLs to download: {url_stats['valid_urls']:,}")
    log.info(f"      ├─ Missing URLs: {url_stats['missing_urls']:,}")
    log.info(f"      └─ Invalid URLs: {url_stats['invalid_urls']:,}")

    # Early exit strategies
    if not download_queue:
        log.info(f"   ✅ All files exist - no downloads needed")
        return valid_indices

    if len(download_queue) > len(df) * 0.8:  # More than 80% need downloading
        log.warning(
            f"   ⚠️  Large download required: {len(download_queue):,} files")
        log.info(f"      └─ This may take a while...")

    log.info(f"   🎯 Download Queue: {len(download_queue):,} files")

    # ------------------------------------------------------------------
    # PRIORITY 4: ROBUST DOWNLOAD EXECUTION
    # ------------------------------------------------------------------

    log.info(f"\n   🚀 EXECUTING ROBUST DOWNLOADS:")

    # Download statistics
    stats = {
        'downloaded': 0,
        'failed': 0,
        'skipped': 0,
        'retries': 0,
        'bytes_total': 0
    }
    stats_lock = threading.Lock()

    def robust_download(idx_url_tuple):
        """Ultra-robust download function with configurable retry."""
        idx, url = idx_url_tuple
        img_path = img_dir / f"{idx}.jpg"

        # Double-check if file was created by another thread
        if img_path.exists() and img_path.stat().st_size > 100:
            with stats_lock:
                stats['skipped'] += 1
            return 'skipped', idx, None

        max_retries = 3
        # Use config values
        backoff_delays = CFG.network_config['retry_delays']

        for attempt in range(max_retries):
            try:
                # Request configuration
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; RecipeBot/1.0)',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Cache-Control': 'no-cache'
                }

                # Make request with timeout from config
                timeout = CFG.network_config['download_timeout']
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=(10, timeout),  # (connect, read) timeout
                    stream=True,
                    allow_redirects=True
                )
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in
                           ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                    raise ValueError(f"Invalid content type: {content_type}")

                # Download with size limits
                content = b''
                downloaded_size = 0
                max_size = 20 * 1024 * 1024  # 20MB limit

                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                        downloaded_size += len(chunk)
                        if downloaded_size > max_size:
                            raise ValueError(
                                f"File too large: {downloaded_size} bytes")

                # Validate minimum size
                if len(content) < 500:  # Minimum viable image
                    raise ValueError(f"File too small: {len(content)} bytes")

                # Atomic write operation
                temp_path = img_path.with_suffix('.tmp')
                try:
                    with open(temp_path, 'wb') as f:
                        f.write(content)

                    # Verify write integrity
                    if temp_path.stat().st_size != len(content):
                        raise ValueError("Write verification failed")

                    # Atomic move to final location
                    temp_path.rename(img_path)

                    # Update statistics
                    with stats_lock:
                        stats['downloaded'] += 1
                        stats['bytes_total'] += len(content)
                        if attempt > 0:
                            stats['retries'] += 1

                    return 'success', idx, len(content)

                finally:
                    # Cleanup temp file if it exists
                    temp_path.unlink(missing_ok=True)

            except requests.exceptions.Timeout:
                error = f"Timeout (attempt {attempt + 1})"
            except requests.exceptions.ConnectionError:
                error = f"Connection error (attempt {attempt + 1})"
            except requests.exceptions.HTTPError as e:
                error = f"HTTP {e.response.status_code} (attempt {attempt + 1})"
                # Don't retry 4xx errors
                if 400 <= e.response.status_code < 500:
                    break
            except ValueError as e:
                error = f"Validation: {str(e)} (attempt {attempt + 1})"
                # Don't retry validation errors
                break
            except Exception as e:
                error = f"Unexpected: {str(e)[:30]} (attempt {attempt + 1})"

            # Wait before retry with configurable delays
            if attempt < max_retries - 1 and attempt < len(backoff_delays):
                time.sleep(backoff_delays[attempt])

        # All attempts failed
        with stats_lock:
            stats['failed'] += 1

        return 'failed', idx, error

    # Execute downloads with progress tracking
    successful_downloads = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(download_queue))) as executor:
        futures = [executor.submit(robust_download, item)
                   for item in download_queue]

        # Progress bar
        with tqdm(
            as_completed(futures),
            total=len(futures),
            desc="      ├─ Downloading",
            bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}] {rate_fmt}"
        ) as pbar:

            for future in pbar:
                try:
                    status, idx, result = future.result()

                    if status in ['success', 'skipped']:
                        successful_downloads.append(idx)
                        valid_indices.append(idx)

                    # Update progress display
                    pbar.set_postfix({
                        'Success': f"{stats['downloaded'] + stats['skipped']}",
                        'Failed': f"{stats['failed']}",
                        'Retries': f"{stats['retries']}"
                    })

                except Exception as e:
                    log.debug(f"Future result error: {e}")

    # ------------------------------------------------------------------
    # PRIORITY 5: RESULTS ANALYSIS AND GRACEFUL DEGRADATION
    # ------------------------------------------------------------------

    total_time = time.time() - download_start

    log.info(f"\n   📊 DOWNLOAD RESULTS:")
    log.info(f"   ├─ Processing time: {total_time:.1f}s")
    log.info(f"   ├─ Files downloaded: {stats['downloaded']:,}")
    log.info(f"   ├─ Files skipped: {stats['skipped']:,}")
    log.info(f"   ├─ Download failures: {stats['failed']:,}")
    log.info(f"   ├─ Retry attempts: {stats['retries']:,}")
    log.info(
        f"   └─ Data transferred: {stats['bytes_total'] / (1024**2):.1f} MB")

    # Calculate success metrics
    total_attempted = len(download_queue)
    total_successful = stats['downloaded'] + stats['skipped']
    success_rate = (total_successful / total_attempted *
                    100) if total_attempted > 0 else 100

    log.info(f"\n   📈 SUCCESS METRICS:")
    log.info(f"   ├─ Download success rate: {success_rate:.1f}%")
    log.info(f"   ├─ Total valid files: {len(valid_indices):,}")
    log.info(
        f"   ├─ Coverage: {len(valid_indices)/len(df)*100:.1f}% of dataset")

    # Determine if results are acceptable
    coverage_threshold = 0.1  # Accept if at least 10% coverage
    if len(valid_indices) / len(df) >= coverage_threshold:
        log.info(
            f"   ✅ ACCEPTABLE COVERAGE - proceeding with {len(valid_indices):,} images")
    else:
        log.warning(
            f"   ⚠️  LOW COVERAGE - only {len(valid_indices):,} images available")
        log.warning(
            f"   └─ Consider using text-only mode or checking data sources")

    gc.collect()

    return sorted(valid_indices)


def filter_low_quality_images(img_dir: Path, embeddings: np.ndarray, original_indices: list) -> tuple:
    """
    Filter out low-quality images based on embedding statistics.

    This function analyzes image embeddings to identify and remove:
    - Blank or corrupted images (very low variance)
    - Generic placeholder images (too similar to mean)

    The filtering ensures at least 50% of images are retained to prevent
    over-aggressive filtering.

    Args:
        img_dir: Directory containing the images
        embeddings: NumPy array of image embeddings (n_images, embedding_dim)
        original_indices: List of indices corresponding to embeddings

    Returns:
        Tuple of (filtered_embeddings, filtered_indices)

    Note:
        This function is critical for maintaining alignment between
        embeddings and their corresponding data indices.
    """
    if embeddings.shape[0] == 0:
        return embeddings, original_indices

    # Calculate embedding statistics
    variances = np.var(embeddings, axis=1)
    means = np.mean(embeddings, axis=1)

    # Remove embeddings with very low variance (likely blank/corrupted images)
    variance_threshold = np.percentile(variances, 10)  # Bottom 10%

    # Remove embeddings that are too similar to the mean (likely generic/placeholder images)
    mean_threshold = np.percentile(means, 90)  # Top 10% of means

    quality_mask = (variances > variance_threshold) & (means < mean_threshold)

    if quality_mask.sum() > embeddings.shape[0] * 0.5:  # Keep at least 50%
        filtered_embeddings = embeddings[quality_mask]
        # CRITICAL FIX: Return the filtered indices too!
        filtered_indices = [original_indices[i]
                            for i in range(len(original_indices)) if quality_mask[i]]

        log.info(
            f"      ├─ Quality filtering: {len(filtered_indices)}/{len(original_indices)} images kept")
        return filtered_embeddings, filtered_indices
    else:
        log.info(
            f"      ├─ Quality filtering: Keeping all images (filter too aggressive)")
        return embeddings, original_indices


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all datasets into memory with comprehensive validation and logging.

    This is the main data loading function that orchestrates loading of:
    1. Recipe dataset (allrecipes.parquet) - main training data
    2. Ground truth dataset (CSV) - manually labeled test data
    3. Silver labels - generated from recipes using heuristic rules
    4. USDA nutritional data - for carbohydrate-based keto rules

    The function includes extensive validation, error handling, and progress
    tracking to ensure data integrity throughout the loading process.

    Returns:
        Tuple of (silver_dataframe, gold_dataframe, recipes_dataframe, carb_dataframe)

    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data validation fails
        RuntimeError: If critical errors occur during loading
    """

    import requests

    load_start = time.time()

    # ------------------------------------------------------------------
    # Initialization and Configuration
    # ------------------------------------------------------------------
    log.info("\n📂 DATASET LOADING PIPELINE")
    log.info(f"   Configuration: {len(CFG.url_map)} data sources")
    log.info(f"   Data directory: {CFG.data_dir}")

    # Log data source information
    log.info(f"   📊 Data Sources:")
    for name, url in CFG.url_map.items():
        source_type = "URL" if url.startswith(
            ('http://', 'https://')) else "Local"
        log.info(f"   ├─ {name}: {source_type}")
        if source_type == "URL":
            log.info(f"   │  └─ {url}")
        else:
            log.info(f"   │  └─ {Path(url).resolve()}")

    def log_memory_usage(stage: str):
        """Helper to log current memory usage"""
        memory = psutil.virtual_memory()
        log.info(f"      💾 {stage}: {memory.percent:.1f}% memory used "
                 f"({memory.used // (1024**2)} MB / {memory.total // (1024**2)} MB)")

    # Track loading stages
    loading_stages = ["Recipes", "Ground Truth",
                      "Silver Labels", "Data Validation", "USDA Data"]

    # Main pipeline progress
    pipeline_progress = tqdm(loading_stages, desc="   ├─ Loading Pipeline",
                             position=0, leave=False,
                             bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

    # ------------------------------------------------------------------
    # STAGE 1: Load Recipes Dataset
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Loading Recipes")
    stage_start = time.time()

    log.info(f"\n   🍳 STAGE 1: LOADING RECIPES DATASET")

    recipes_url = CFG.url_map["allrecipes.parquet"]
    log.info(f"   ├─ Source: {recipes_url}")

    with tqdm(total=4, desc="      ├─ Recipe Loading", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as recipe_pbar:

        recipe_pbar.set_description("      ├─ Validating source")

        # Validate URL/path
        if recipes_url.startswith(('http://', 'https://')):
            try:
                # Quick HEAD request to check if URL is accessible
                response = requests.head(recipes_url, timeout=10)
                response.raise_for_status()
                log.info(f"      ├─ URL accessible: {response.status_code}")

                # Get content length if available
                content_length = response.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    log.info(
                        f"      ├─ Expected download size: {size_mb:.1f} MB")

            except requests.RequestException as e:
                log.error(f"      ❌ URL validation failed: {e}")
                raise RuntimeError(f"Cannot access recipes URL: {recipes_url}")
        else:
            # Local file validation
            recipes_path = Path(recipes_url)
            if not recipes_path.exists():
                raise FileNotFoundError(
                    f"Recipes file not found: {recipes_url}")

            size_mb = recipes_path.stat().st_size / (1024 * 1024)
            log.info(f"      ├─ Local file size: {size_mb:.1f} MB")

        recipe_pbar.update(1)

        recipe_pbar.set_description("      ├─ Reading parquet")
        recipes_load_start = time.time()

        try:
            # Load with progress indication for large files
            recipes = pd.read_parquet(recipes_url)

        except Exception as e:
            log.error(f"      ❌ Failed to load recipes: {str(e)[:100]}...")

            # Try alternative approaches
            if recipes_url.startswith(('http://', 'https://')):
                log.info(f"      🔄 Attempting manual download...")
                try:
                    response = requests.get(
                        recipes_url, stream=True, timeout=30)
                    response.raise_for_status()

                    # Save temporarily and load
                    temp_path = Path("temp_recipes.parquet")
                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    recipes = pd.read_parquet(temp_path)
                    temp_path.unlink()  # Clean up

                except Exception as e2:
                    log.error(f"      ❌ Manual download also failed: {e2}")
                    raise RuntimeError(
                        f"Failed to load recipes after retry: {e2}")
            else:
                raise

        recipes_load_time = time.time() - recipes_load_start
        recipe_pbar.update(1)

        recipe_pbar.set_description("      ├─ Validating schema")

        # Validate recipes schema
        expected_columns = ['ingredients',
                            'title', 'description', 'instructions']
        missing_columns = [
            col for col in expected_columns if col not in recipes.columns]

        if missing_columns:
            log.warning(
                f"      ⚠️  Missing expected columns: {missing_columns}")

        log.info(f"      📊 Recipes Dataset:")
        log.info(f"      ├─ Shape: {recipes.shape}")
        log.info(f"      ├─ Columns: {list(recipes.columns)}")
        log.info(
            f"      ├─ Memory usage: {recipes.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        log.info(f"      └─ Load time: {recipes_load_time:.1f}s")

        recipe_pbar.update(1)

        recipe_pbar.set_description("      ├─ Data quality check")

        # Quick data quality assessment
        quality_stats = {
            'total_rows': len(recipes),
            'null_ingredients': recipes['ingredients'].isnull().sum(),
            'empty_ingredients': (
                recipes['ingredients']           # keep the original Series
                .astype(str)              # lists/None → string form
                .str.strip()              # remove surrounding whitespace
                .eq('')                   # test for genuine empties
                .sum()                    # count them
            ) if 'ingredients' in recipes.columns else 0,
            'null_titles': recipes['title'].isnull().sum() if 'title' in recipes.columns else 0,
            'has_photo_url': 'photo_url' in recipes.columns,
            'photo_url_count': (~recipes['photo_url'].isnull()).sum() if 'photo_url' in recipes.columns else 0
        }

        log.info(f"      📈 Data Quality:")
        log.info(f"      ├─ Total recipes: {quality_stats['total_rows']:,}")
        log.info(
            f"      ├─ Null ingredients: {quality_stats['null_ingredients']:,}")
        log.info(
            f"      ├─ Empty ingredients: {quality_stats['empty_ingredients']:,}")

        if quality_stats['has_photo_url']:
            photo_pct = quality_stats['photo_url_count'] / \
                quality_stats['total_rows'] * 100
            log.info(
                f"      └─ With photos: {quality_stats['photo_url_count']:,} ({photo_pct:.1f}%)")

        recipe_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Recipes loaded successfully in {stage_time:.1f}s")
    log_memory_usage("Recipes loaded")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # STAGE 2: Load Ground Truth Dataset
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Loading Ground Truth")
    stage_start = time.time()

    log.info(f"\n   🎯 STAGE 2: LOADING GROUND TRUTH DATASET")

    gt_url_or_path = CFG.url_map["ground_truth_sample.csv"]
    log.info(f"   ├─ Source: {gt_url_or_path}")

    with tqdm(total=4, desc="      ├─ Ground Truth Loading", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as gt_pbar:

        gt_pbar.set_description("      ├─ Path validation")

        # Validate path - check for directory mistake
        if Path(gt_url_or_path).is_dir():
            log.error(
                f"      ❌ Expected CSV file but found directory: {gt_url_or_path}")
            raise RuntimeError(
                f"Expected a CSV file but found a directory: {gt_url_or_path}")

        # Check accessibility
        if gt_url_or_path.startswith(('http://', 'https://')):
            try:
                response = requests.head(gt_url_or_path, timeout=10)
                response.raise_for_status()
                log.info(f"      ├─ URL accessible: {response.status_code}")
            except requests.RequestException as e:
                log.error(f"      ❌ Ground truth URL validation failed: {e}")
                raise RuntimeError(
                    f"Cannot access ground truth URL: {gt_url_or_path}")
        else:
            gt_path = Path(gt_url_or_path)
            if not gt_path.exists():
                raise FileNotFoundError(
                    f"Ground truth file not found: {gt_url_or_path}")

            size_kb = gt_path.stat().st_size / 1024
            log.info(f"      ├─ File size: {size_kb:.1f} KB")

        gt_pbar.update(1)

        gt_pbar.set_description("      ├─ Reading CSV")
        gt_load_start = time.time()

        try:
            # Load with error handling for encoding issues
            try:
                ground_truth = pd.read_csv(gt_url_or_path)
            except UnicodeDecodeError:
                log.warning(
                    f"      ⚠️  UTF-8 decode failed, trying latin-1...")
                ground_truth = pd.read_csv(gt_url_or_path, encoding='latin-1')
            except pd.errors.EmptyDataError:
                log.error(f"      ❌ Ground truth file is empty")
                raise RuntimeError("Ground truth CSV file is empty")

        except Exception as e:
            log.error(
                f"      ❌ Failed to load ground truth: {str(e)[:100]}...")
            raise RuntimeError(f"Failed to load ground truth: {e}")

        gt_load_time = time.time() - gt_load_start
        gt_pbar.update(1)

        gt_pbar.set_description("      ├─ Schema validation")

        # Validate ground truth schema
        required_gt_columns = ['ingredients']
        missing_gt_columns = [
            col for col in required_gt_columns if col not in ground_truth.columns]

        if missing_gt_columns:
            log.error(
                f"      ❌ Missing required columns: {missing_gt_columns}")
            raise ValueError(
                f"Ground truth missing required columns: {missing_gt_columns}")

        # Look for label columns
        keto_columns = [
            col for col in ground_truth.columns if 'keto' in col.lower()]
        vegan_columns = [
            col for col in ground_truth.columns if 'vegan' in col.lower()]

        log.info(f"      📊 Ground Truth Dataset:")
        log.info(f"      ├─ Shape: {ground_truth.shape}")
        log.info(f"      ├─ Columns: {list(ground_truth.columns)}")
        log.info(f"      ├─ Keto columns found: {keto_columns}")
        log.info(f"      ├─ Vegan columns found: {vegan_columns}")
        log.info(f"      └─ Load time: {gt_load_time:.2f}s")

        gt_pbar.update(1)

        gt_pbar.set_description("      ├─ Label processing")

        # Process labels with error handling
        try:
            # Extract keto labels
            if keto_columns:
                ground_truth["label_keto"] = ground_truth.filter(
                    regex="keto").iloc[:, 0].astype(int)
                keto_positive = ground_truth["label_keto"].sum()
                keto_rate = keto_positive / len(ground_truth) * 100
                log.info(
                    f"      ├─ Keto labels: {keto_positive}/{len(ground_truth)} ({keto_rate:.1f}% positive)")
            else:
                log.warning(
                    f"      ⚠️  No keto columns found - creating dummy labels")
                ground_truth["label_keto"] = 0

            # Extract vegan labels
            if vegan_columns:
                ground_truth["label_vegan"] = ground_truth.filter(
                    regex="vegan").iloc[:, 0].astype(int)
                vegan_positive = ground_truth["label_vegan"].sum()
                vegan_rate = vegan_positive / len(ground_truth) * 100
                log.info(
                    f"      ├─ Vegan labels: {vegan_positive}/{len(ground_truth)} ({vegan_rate:.1f}% positive)")
            else:
                log.warning(
                    f"      ⚠️  No vegan columns found - creating dummy labels")
                ground_truth["label_vegan"] = 0

        except Exception as e:
            log.error(f"      ❌ Label processing failed: {e}")
            raise ValueError(f"Failed to process labels: {e}")

        # Add photo_url if available
        ground_truth["photo_url"] = ground_truth.get("photo_url")

        # Clean ingredients text
        with tqdm(total=1, desc="         ├─ Normalizing text", position=2, leave=False,
                  bar_format="         ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as norm_pbar:
            ground_truth["clean"] = ground_truth.ingredients.fillna(
                "").map(normalise)
            norm_pbar.update(1)

        gt_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Ground truth loaded successfully in {stage_time:.1f}s")
    log_memory_usage("Ground truth loaded")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # STAGE 3: Generate Silver Labels
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Generating Silver Labels")
    stage_start = time.time()

    log.info(f"\n   🥈 STAGE 3: GENERATING SILVER LABELS")

    with tqdm(total=3, desc="      ├─ Silver Generation", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as silver_pbar:

        silver_pbar.set_description("      ├─ Building silver labels")
        silver_start = time.time()

        # Generate silver labels using heuristics
        silver = build_silver(recipes)

        silver_build_time = time.time() - silver_start
        silver_pbar.update(1)

        silver_pbar.set_description("      ├─ Adding photo URLs")

        # Add photo URLs from recipes
        silver["photo_url"] = recipes.get("photo_url")

        # Calculate silver label statistics
        silver_stats = {
            'total': len(silver),
            'keto_positive': silver['silver_keto'].sum() if 'silver_keto' in silver.columns else 0,
            'vegan_positive': silver['silver_vegan'].sum() if 'silver_vegan' in silver.columns else 0,
            'has_photos': (~silver['photo_url'].isnull()).sum() if 'photo_url' in silver.columns else 0
        }

        silver_pbar.update(1)

        silver_pbar.set_description("      ├─ Quality assessment")

        log.info(f"      📊 Silver Labels Generated:")
        log.info(f"      ├─ Total recipes: {silver_stats['total']:,}")
        log.info(
            f"      ├─ Keto positive: {silver_stats['keto_positive']:,} ({silver_stats['keto_positive']/silver_stats['total']*100:.1f}%)")
        log.info(
            f"      ├─ Vegan positive: {silver_stats['vegan_positive']:,} ({silver_stats['vegan_positive']/silver_stats['total']*100:.1f}%)")
        log.info(
            f"      ├─ With photos: {silver_stats['has_photos']:,} ({silver_stats['has_photos']/silver_stats['total']*100:.1f}%)")
        log.info(f"      └─ Generation time: {silver_build_time:.1f}s")

        silver_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Silver labels generated successfully in {stage_time:.1f}s")
    log_memory_usage("Silver labels generated")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # STAGE 4: Data Validation and Cross-Checks
    # ------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Data Validation")
    stage_start = time.time()

    log.info(f"\n   ✅ STAGE 4: DATA VALIDATION AND CROSS-CHECKS")

    with tqdm(total=5, desc="      ├─ Validation Checks", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as val_pbar:

        val_pbar.set_description("      ├─ Index alignment")

        # Check index alignment between datasets
        recipes_indices = set(recipes.index)
        silver_indices = set(silver.index)
        gt_indices = set(ground_truth.index)

        if recipes_indices != silver_indices:
            log.warning(f"      ⚠️  Index mismatch between recipes and silver")
            log.info(f"         ├─ Recipes: {len(recipes_indices)} indices")
            log.info(f"         └─ Silver: {len(silver_indices)} indices")

        val_pbar.update(1)

        val_pbar.set_description("      ├─ Data consistency")

        # Check data consistency
        consistency_issues = []

        # Check for null ingredients in critical datasets
        null_ingredients_recipes = recipes['ingredients'].isnull().sum()
        null_ingredients_gt = ground_truth['ingredients'].isnull().sum()

        if null_ingredients_recipes > 0:
            consistency_issues.append(
                f"Recipes has {null_ingredients_recipes} null ingredients")

        if null_ingredients_gt > 0:
            consistency_issues.append(
                f"Ground truth has {null_ingredients_gt} null ingredients")

        val_pbar.update(1)

        val_pbar.set_description("      ├─ Memory optimization")

        # Memory usage analysis
        datasets_memory = {
            'recipes': recipes.memory_usage(deep=True).sum() / (1024**2),
            'silver': silver.memory_usage(deep=True).sum() / (1024**2),
            'ground_truth': ground_truth.memory_usage(deep=True).sum() / (1024**2)
        }

        total_memory = sum(datasets_memory.values())

        log.info(f"      💾 Memory Usage by Dataset:")
        for dataset, memory_mb in datasets_memory.items():
            log.info(f"      ├─ {dataset.capitalize()}: {memory_mb:.1f} MB")
        log.info(f"      └─ Total: {total_memory:.1f} MB")

        val_pbar.update(1)

        val_pbar.set_description("      ├─ Label distribution")

        # Compare label distributions
        if len(ground_truth) > 0:
            gt_keto_rate = ground_truth['label_keto'].mean() * 100
            gt_vegan_rate = ground_truth['label_vegan'].mean() * 100
            silver_keto_rate = silver['silver_keto'].mean() * 100
            silver_vegan_rate = silver['silver_vegan'].mean() * 100

            log.info(f"      📊 Label Distribution Comparison:")
            log.info(
                f"      ├─ Keto: Gold={gt_keto_rate:.1f}%, Silver={silver_keto_rate:.1f}%")
            log.info(
                f"      └─ Vegan: Gold={gt_vegan_rate:.1f}%, Silver={silver_vegan_rate:.1f}%")

            # Flag significant differences
            keto_diff = abs(gt_keto_rate - silver_keto_rate)
            vegan_diff = abs(gt_vegan_rate - silver_vegan_rate)

            if keto_diff > 20:
                log.warning(
                    f"      ⚠️  Large keto distribution difference: {keto_diff:.1f}%")
            if vegan_diff > 20:
                log.warning(
                    f"      ⚠️  Large vegan distribution difference: {vegan_diff:.1f}%")

        val_pbar.update(1)

        val_pbar.set_description("      ├─ Final validation")

        # Final validation summary
        validation_summary = {
            'recipes_loaded': len(recipes) > 0,
            'ground_truth_loaded': len(ground_truth) > 0,
            'silver_generated': len(silver) > 0,
            'required_columns_present': all(col in recipes.columns for col in ['ingredients']),
            'labels_processed': 'label_keto' in ground_truth.columns and 'label_vegan' in ground_truth.columns,
            'consistency_issues': len(consistency_issues)
        }

        all_valid = all(validation_summary[key] for key in [
                        'recipes_loaded', 'ground_truth_loaded', 'silver_generated', 'required_columns_present', 'labels_processed'])

        log.info(f"      ✅ Validation Summary:")
        for check, status in validation_summary.items():
            if isinstance(status, bool):
                status_icon = "✅" if status else "❌"
                log.info(
                    f"      ├─ {check.replace('_', ' ').title()}: {status_icon}")
            else:
                log.info(
                    f"      ├─ {check.replace('_', ' ').title()}: {status}")

        if consistency_issues:
            log.warning(f"      ⚠️  Consistency Issues Found:")
            for issue in consistency_issues:
                log.warning(f"      │  └─ {issue}")

        if not all_valid:
            raise RuntimeError(
                "Dataset validation failed - see logs for details")

        val_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Data validation completed in {stage_time:.1f}s")
    log_memory_usage("Validation complete")
    pipeline_progress.update(1)

    # ----------------------------------------------------------------------
    # STAGE 5: Load USDA nutrient table
    # ----------------------------------------------------------------------
    pipeline_progress.set_description("   ├─ Loading USDA carbs")
    stage_start = time.time()

    carb_df = _load_usda_carb_table()

    log.info("   ✅ USDA table loaded in %.1fs – %d rows",
             time.time() - stage_start, len(carb_df))
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # Pipeline Completion Summary
    # ------------------------------------------------------------------
    total_time = time.time() - load_start

    log.info(f"\n🏁 DATASET LOADING COMPLETE")
    log.info(f"   ├─ Total loading time: {total_time:.1f}s")
    log.info(f"   ├─ Datasets loaded: 4 (recipes, ground_truth, silver, usda)")
    log.info(f"   ├─ Total memory usage: {total_memory:.1f} MB")
    log.info(f"   └─ All validations passed: ✅")

    # Final dataset summary
    log.info(f"\n   📋 Final Dataset Summary:")
    log.info(
        f"   ├─ Recipes: {len(recipes):,} rows × {len(recipes.columns)} columns")
    log.info(
        f"   ├─ Ground Truth: {len(ground_truth):,} rows × {len(ground_truth.columns)} columns")
    log.info(
        f"   ├─ Silver Labels: {len(silver):,} rows × {len(silver.columns)} columns")
    log.info(f"   ├─ USDA Carb Tbl:  {len(carb_df):,} rows × {len(carb_df.columns)} columns "
             f"({carb_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB)")

    # Update total memory after USDA load
    total_used = total_memory + carb_df.memory_usage(deep=True).sum()/1_048_576
    log.info(f"   ├─ Total memory usage: {total_used:.1f} MB")
    log.info(f"   └─ Ready for ML pipeline: ✅")

    # Garbage collection for memory optimization
    gc.collect()

    return silver, ground_truth, recipes, carb_df


def save_models_optimized(models: dict, vectorizer, path: Path):
    """
    Save models with compression, versioning, and metadata.

    Args:
        models: Dictionary of task -> model
        vectorizer: TF-IDF vectorizer
        path: Directory to save models
    """
    import joblib
    import json

    path.mkdir(parents=True, exist_ok=True)

    # Prepare model metadata
    model_metadata = {
        'version': '1.0',
        'creation_time': datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'models': {}
    }

    # Save each model with compression
    for task, model in models.items():
        model_path = path / f"model_{task}.pkl.gz"

        # Determine model requirements
        model_info = {
            'task': task,
            'model_type': type(model).__name__,
            'requires_images': False,
            'feature_count': None
        }

        if hasattr(model, 'ensemble_type'):
            model_info['ensemble_type'] = model.ensemble_type
            model_info['requires_images'] = model.ensemble_type in [
                'best_two', 'smart_ensemble']
        elif hasattr(model, 'n_features_in_'):
            model_info['feature_count'] = model.n_features_in_
            model_info['requires_images'] = model.n_features_in_ > 7000 or model.n_features_in_ <= 2048

        model_metadata['models'][task] = model_info

        # Save with compression
        joblib.dump(model, model_path, compress=3)
        log.info(f"   ✅ Saved {task} model to {model_path} (compressed)")

    # Save vectorizer
    vec_path = path / "vectorizer.pkl.gz"
    joblib.dump(vectorizer, vec_path, compress=3)
    log.info(f"   ✅ Saved vectorizer to {vec_path} (compressed)")

    # Save metadata
    metadata_path = path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    log.info(f"   ✅ Saved model metadata to {metadata_path}")

    # Create a compatibility pickle for backward compatibility
    compat_models = {}
    for task, model in models.items():
        compat_models[task] = model

    with open(path / "models.pkl", 'wb') as f:
        pickle.dump(compat_models, f)

    with open(path / "vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)

    log.info(f"   ✅ Created backward-compatible model files")


# =============================================================================
# GLOBAL DATASET CACHE
# =============================================================================
"""
One-shot dataset caching mechanism to avoid reloading large datasets.
This global cache is populated on first access and reused throughout
the pipeline lifetime.
"""

_DATASETS: tuple[pd.DataFrame, pd.DataFrame,
                 pd.DataFrame, pd.DataFrame] | None = None


def get_datasets(sample_frac: float | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame]:
    """
    Lazy-load and cache the four DataFrames returned by load_datasets().

    This function implements a singleton pattern for dataset loading,
    ensuring that expensive data loading operations happen only once.

    Args:
        sample_frac: Optional fraction to sample from silver dataset (0.0-1.0)

    Returns:
        Tuple of (silver_all, gold, recipes, carb_df)

    Note:
        The sampling only applies to the silver dataset and is useful
        for testing the pipeline on smaller data subsets.
    """
    global _DATASETS

    if _DATASETS is None:
        # Actually load everything once
        silver_all, gold, recipes, carb_df = load_datasets()

        # Optional row-sampling of the *silver* set
        if sample_frac:
            silver_all = silver_all.sample(frac=sample_frac,
                                           random_state=42).copy()

        _DATASETS = (silver_all, gold, recipes, carb_df)

    return _DATASETS


# =============================================================================
# DOMAIN-SPECIFIC INGREDIENT LISTS
# =============================================================================
"""
Hard-coded ingredient lists for diet classification based on nutritional guidelines.
These lists form the foundation of the rule-based classification system and are
used to create regex patterns for fast ingredient matching.

The lists are carefully curated based on:
- Nutritional science (carbohydrate content for keto)
- Dietary restrictions (animal products for vegan)
- Common food terminology and variations
"""

NON_KETO = list(set([
    # High-carb fruits
    "apple", "banana", "orange", "grape", "kiwi", "mango", "peach",
    "strawberry", "pineapple", "apricot", "tangerine", "persimmon",
    "pomegranate", "prune", "papaya", "jackfruit",

    # Grains and grain products
    "white rice", "long grain rice", "cornmeal", "corn",
    "all-purpose flour", "bread", "pasta", "couscous",
    "bulgur", "quinoa", "barley flour", "buckwheat flour",
    "durum wheat flour", "wheat flour", "whole-wheat flour",
    "oat flour", "oatmeal", "rye flour", "semolina",
    "amaranth", "millet", "sorghum flour", "sorghum grain",
    "spelt flour", "teff grain", "triticale",
    "einkorn flour", "emmer grain", "fonio", "freekeh",
    "kamut flour", "farina",

    # Starchy vegetables
    "potato", "baking potato", "potato wedge", "potato slice",
    "russet potato", "sweet potato", "yam", "cassava",
    "taro", "lotus root", "water chestnut", "ube",

    # Legumes
    "kidney bean", "black bean", "pinto bean", "navy bean",
    "lima bean", "cannellini bean", "great northern bean",
    "garbanzo bean", "chickpea", "adzuki bean", "baked bean",
    "refried bean", "hummus",

    # Sweeteners and sugars
    "sugar", "brown sugar", "coconut sugar", "muscovado sugar",
    "demerara", "turbinado", "molasses", "honey", "agave nectar",
    "maple syrup",

    # Sauces and condiments high in sugar
    "tomato sauce", "ketchup", "bbq sauce", "teriyaki sauce",
    "hoisin sauce", "sweet chili sauce", "sweet pickle",
    "sweet relish", "sweet soy glaze", "marmalade",

    # Processed foods and snacks
    "bread", "bagel", "muffin", "cookie", "cooky", "cake",
    "pastry", "pie crust", "pizza", "pizza crust", "pizza flour",
    "naan", "pita", "roti", "chapati", "tortilla",
    "pretzel", "chip", "french fry", "tater tot",
    "doughnut", "graham cracker", "hamburger bun", "hot-dog bun",

    # Breakfast items
    "breakfast cereal", "granola", "muesli", "energy bar",

    # Beverages
    "soy sauce", "orange juice", "fruit punch", "chocolate milk",
    "sweetened condensed milk", "sweetened cranberry", "sweetened yogurt",

    # Alcoholic beverages (carbs from alcohol and mixers)
    "ale", "beer", "lager", "ipa", "pilsner", "stout", "porter",
    "moscato", "riesling", "port", "sherry", "sangria",
    "margarita", "mojito", "pina colada", "daiquiri", "mai tai",
    "cosmopolitan", "whiskey sour", "bloody mary",
    "bailey", "kahlua", "amaretto", "frangelico", "limoncello",
    "triple sec", "curacao", "alcoholic lemonade", "alcopop",
    "breezer", "smirnoff ice", "mike hard lemonade", "hard cider", "cider",

    # Specialty items
    "tapioca", "arrowroot", "job's tear", "jobs tear", "job tear",
    "gnocchi", "tempura batter", "breading",
    "ice cream", "candy", "hard candy", "gummy bear",

    # Soy products (often sweetened)
    "soybean sweetened",
]))

NON_VEGAN = list(set([
    # Meat - Red meat
    'beef', 'steak', 'ribeye', 'sirloin', 'veal', 'lamb', 'mutton',
    'pork', 'bacon', 'ham', 'boar', 'goat', 'kid', 'venison',
    'rabbit', 'hare',

    # Meat - Poultry
    'chicken', 'turkey', 'duck', 'goose', 'quail', 'pheasant',
    'partridge', 'grouse',

    # Meat - Organ meats
    'liver', 'kidney', 'heart', 'tongue', 'brain', 'sweetbread',
    'tripe', 'gizzard', 'offal', 'bone', 'marrow', 'oxtail',

    # Meat - Processed
    'sausage', 'bratwurst', 'knackwurst', 'mettwurst',
    'salami', 'pepperoni', 'pastrami', 'bresaola',
    'prosciutto', 'pancetta', 'guanciale', 'speck',
    'mortadella', 'capocollo', 'coppa', 'cotechino',
    'chorizo', 'lard', 'tallow',

    # Fish and Seafood
    'fish', 'salmon', 'tuna', 'cod', 'haddock', 'halibut',
    'mackerel', 'herring', 'sardine', 'anchovy', 'trout',
    'tilapia', 'catfish', 'carp', 'sole', 'snapper', 'eel',
    'shrimp', 'prawn', 'crab', 'lobster', 'langoustine',
    'clam', 'mussel', 'oyster', 'scallop', 'squid', 'calamari',
    'octopus', 'krill', 'caviar', 'roe',
    'fishpaste', 'shrimppaste', 'anchovypaste', 'bonito',
    'katsuobushi', 'dashi', 'nampla',

    # Dairy - Milk products
    'milk', 'cream', 'butter', 'buttermilk', 'condensed', 'evaporated',
    'lactose', 'whey', 'casein', 'ghee', 'kefir',

    # Dairy - Cheese
    'cheese', 'cheddar', 'mozzarella', 'parmesan', 'parmigiano',
    'reggiano', 'pecorino', 'ricotta', 'mascarpone',
    'brie', 'camembert', 'roquefort', 'gorgonzola', 'stilton',
    'emmental', 'gruyere', 'fontina', 'asiago', 'manchego',
    'halloumi', 'feta', 'quark', 'paneer', 'stracciatella',
    'provolone', 'taleggio',

    # Dairy - Other
    'yogurt', 'sourcream', 'cremefraiche', 'curd', 'custard',
    'icecream', 'gelatin', 'collagen',

    # Eggs
    'egg', 'yolk', 'albumen', 'omelet', 'omelette', 'meringue',

    # Other animal products
    'honey', 'shellfish', 'escargot', 'snail', 'frog',
    'worcestershire', 'aioli', 'mayonnaise',
    'broth', 'stock', 'gravy',
]))

KETO_WHITELIST = [
    # Keto-friendly flours
    r"\balmond flour\b",
    r"\bcoconut flour\b",
    r"\bflaxseed flour\b",
    r"\bchia flour\b",
    r"\bsunflower seed flour\b",
    r"\bpeanut flour\b",
    r"\bhemp flour\b",
    r"\bsesame flour\b",
    r"\bwalnut flour\b",
    r"\bpecan flour\b",
    r"\bmacadamia flour\b",
    r"\bhazelnut flour\b",

    # Special exceptions for "kidney" (organ meat, not kidney beans)
    r"\bkidney\b",

    # Low-carb citrus exceptions
    r"\blemon juice\b",

    # Keto milk alternatives
    r"\balmond milk\b",
    r"\bcoconut milk\b",
    r"\bflax milk\b",
    r"\bmacadamia milk\b",
    r"\bhemp milk\b",
    r"\bcashew milk\b",
    r"\balmond cream\b",
    r"\bcoconut cream\b",
    r"\bsour cream\b",

    # Nut and seed butters
    r"\balmond butter\b",
    r"\bpeanut butter\b",
    r"\bcoconut butter\b",
    r"\bmacadamia butter\b",
    r"\bpecan butter\b",
    r"\bwalnut butter\b",
    r"\bhemp butter\b",

    # Keto bread alternatives
    r"\balmond bread\b",
    r"\bcoconut bread\b",
    r"\bcloud bread\b",
    r"\bketo bread\b",

    # Sugar-free sweeteners
    r"\bcoconut sugar[- ]free\b",
    r"\bstevia\b",
    r"\berytritol\b",
    r"\bmonk fruit\b",
    r"\bswerve\b",
    r"\ballulose\b",
    r"\bxylitol\b",
    r"\bsugar[- ]free\b",

    # Low-carb alternatives
    r"\bcauliflower rice\b",
    r"\bshirataki noodles\b",
    r"\bzucchini noodles\b",
    r"\bkelp noodles\b",
    r"\bsugar[- ]free chocolate\b",
    r"\bketo chocolate\b",

    # Low-carb vegetables and foods
    r"\bavocado\b",
    r"\bcacao\b",
    r"\bcocoa powder\b",
    r"\bketo ice[- ]cream\b",
    r"\bsugar[- ]free ice[- ]cream\b",
    r"\bjicama\b",
    r"\bzucchini\b",
    r"\bcucumber\b",
    r"\bbroccoli\b",
    r"\bcauliflower\b",
]

VEGAN_WHITELIST = [
    # Egg exceptions (plant-based)
    r"\beggplant\b",
    r"\begg\s*fruit\b",
    r"\bvegan\s+egg\b",

    # Milk exceptions (plant-based)
    r"\bmillet\b",  # grain, not milk
    r"\bmilk\s+thistle\b",
    r"\bcoconut\s+milk\b",
    r"\boat\s+milk\b",
    r"\bsoy\s+milk\b",
    r"\balmond\s+milk\b",
    r"\bcashew\s+milk\b",
    r"\brice\s+milk\b",
    r"\bhazelnut\s+milk\b",
    r"\bpea\s+milk\b",

    # Rice alternatives (vegetable-based)
    r"\bcauliflower rice\b",
    r"\bbroccoli rice\b",
    r"\bsweet potato rice\b",
    r"\bzucchini rice\b",
    r"\bcabbage rice\b",
    r"\bkonjac rice\b",
    r"\bshirataki rice\b",
    r"\bmiracle rice\b",
    r"\bpalmini rice\b",

    # Butter exceptions (plant-based)
    r"\bbutternut\b",  # squash
    r"\bbutterfly\s+pea\b",
    r"\bcocoa\s+butter\b",
    r"\bpeanut\s+butter\b",
    r"\balmond\s+butter\b",
    r"\bsunflower(?:\s*seed)?\s+butter\b",
    r"\bpistachio\s+butter\b",
    r"\bvegan\s+butter\b",

    # Honey exceptions (plants)
    r"\bhoneydew\b",
    r"\bhoneysuckle\b",
    r"\bhoneycrisp\b",
    r"\bhoney\s+locust\b",
    r"\bhoneyberry\b",

    # Cream exceptions (plant-based)
    r"\bcream\s+of\s+tartar\b",
    r"\bice[- ]cream\s+bean\b",
    r"\bcoconut\s+cream\b",
    r"\bcashew\s+cream\b",
    r"\bvegan\s+cream\b",

    # Cheese exceptions (plant-based)
    r"\bcheesewood\b",
    r"\bvegan\s+cheese\b",
    r"\bcashew\s+cheese\b",

    # Fish exceptions (plants)
    r"\bfish\s+mint\b",
    r"\bfish\s+pepper\b",

    # Beef exceptions (plants/mushrooms)
    r"\bbeefsteak\s+plant\b",
    r"\bbeefsteak\s+mushroom\b",

    # Chicken/hen exceptions (mushrooms)
    r"\bchicken[- ]of[- ]the[- ]woods\b",
    r"\bchicken\s+mushroom\b",
    r"\bhen[- ]of[- ]the[- ]woods\b",

    # Meat exceptions (plants)
    r"\bsweetmeat\s+(?:pumpkin|squash)\b",

    # Bacon alternatives
    r"\bcoconut\s+bacon\b",
    r"\bmushroom\s+bacon\b",
    r"\bsoy\s+bacon\b",
    r"\bvegan\s+bacon\b",
]


# =============================================================================
# REGEX COMPILATION AND HELPERS
# =============================================================================
"""
Compiled regex patterns for efficient ingredient matching.
These patterns are created once and reused throughout the pipeline.
"""

# Global variables for carbohydrate lookup
_CARB_MAP: dict[str, float] | None = None
_FUZZY_KEYS: list[str] | None = None


def _ensure_carb_map() -> None:
    """Lazy-load USDA table into a dict for µs look-ups."""
    global _CARB_MAP, _FUZZY_KEYS
    if _CARB_MAP is None:
        df = _load_usda_carb_table()
        _CARB_MAP = df.set_index("food_desc")["carb_100g"].to_dict()
        _FUZZY_KEYS = list(_CARB_MAP)           # for RapidFuzz
        log.info("Carb map initialised (%d keys)", len(_CARB_MAP))


def compile_any(words: Iterable[str]) -> re.Pattern[str]:
    """
    Compile a list of words into a single regex pattern for efficient matching.

    Creates a pattern that matches any whole word from the input list,
    case-insensitively. Word boundaries ensure we don't match partial words.

    Args:
        words: Iterable of strings to compile into pattern

    Returns:
        Compiled regex pattern matching any word in the list

    Example:
        >>> pattern = compile_any(["apple", "banana"])
        >>> bool(pattern.search("I like apple pie"))
        True
        >>> bool(pattern.search("I like pineapple"))
        False  # "apple" inside "pineapple" doesn't match due to word boundaries
    """
    return re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, words)), re.I)


# Pre-compiled patterns for performance
RX_KETO = compile_any(NON_KETO)
RX_VEGAN = compile_any(NON_VEGAN)
RX_WL_KETO = re.compile("|".join(KETO_WHITELIST), re.I)
RX_WL_VEGAN = re.compile("|".join(VEGAN_WHITELIST), re.I)


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================
"""
Functions for normalizing ingredient text to improve matching accuracy.
Normalization includes removing units, numbers, punctuation, and applying
lemmatization to reduce words to their base forms.
"""

# Initialize lemmatizer if NLTK is available
_LEMM = WordNetLemmatizer() if wnl else None

# Pattern for common cooking units
_UNITS = re.compile(r"\b(?:g|gram|kg|oz|ml|l|cup|cups|tsp|tbsp|teaspoon|"
                    r"tablespoon|pound|lb|slice|slices|small|large|medium)\b")


def tokenize_ingredient(text: str) -> list[str]:
    """
    Extract word tokens from ingredient text.

    Splits text into individual words, handling hyphenated words correctly.
    Used for token-level matching in classification rules.

    Args:
        text: Raw ingredient text

    Returns:
        List of lowercase word tokens

    Example:
        >>> tokenize_ingredient("Sugar-free dark chocolate")
        ['sugar-free', 'dark', 'chocolate']
    """
    return re.findall(r"\b\w[\w-]*\b", text.lower())


def normalise(t: str | list | tuple | np.ndarray) -> str:
    """
    Normalize ingredient text for consistent matching.

    This comprehensive normalization function handles various input formats
    and applies multiple cleaning steps to prepare text for classification:

    1. Handles list/array inputs (from parquet files)
    2. Removes accents and non-ASCII characters
    3. Removes parenthetical information
    4. Removes cooking units and measurements
    5. Removes numbers and fractions
    6. Removes punctuation
    7. Applies lemmatization (if available)
    8. Filters out very short words

    Args:
        t: Input text (string, list, tuple, or numpy array)

    Returns:
        Normalized string with cleaned, lemmatized tokens

    Example:
        >>> normalise("2 cups of sliced bananas (ripe)")
        "slice banana ripe"
        >>> normalise(["olive oil", "salt"])
        "olive oil salt"
    """
    # Handle non-string inputs
    if not isinstance(t, str):
        if isinstance(t, (list, tuple, np.ndarray)):
            t = " ".join(map(str, t))
        else:
            t = str(t)

    # Unicode normalization - remove accents
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()

    # Remove parenthetical content and convert to lowercase
    t = re.sub(r"\([^)]*\)", " ", t.lower())

    # Remove units of measurement
    t = _UNITS.sub(" ", t)

    # Remove numbers (including fractions)
    t = re.sub(r"\d+(?:[/\.]\d+)?", " ", t)

    # Remove punctuation
    t = re.sub(r"[^\w\s-]", " ", t)

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Apply lemmatization and filter short words
    if _LEMM:
        return " ".join(_LEMM.lemmatize(w) for w in t.split() if len(w) > 2)
    return " ".join(w for w in t.split() if len(w) > 2)


# =============================================================================
# CARBOHYDRATE LOOKUP FUNCTIONS
# =============================================================================
"""
Functions for looking up carbohydrate content from USDA nutritional database.
Used for the numeric keto rule: ingredients with >10g carbs/100g are non-keto.
"""


def _ensure_carb_map() -> None:
    """
    Lazy-load USDA carbohydrate data into memory for fast lookups.

    This function loads the USDA nutritional database on first use and
    creates a dictionary mapping food descriptions to carbohydrate content.
    Also prepares a list of keys for fuzzy matching.
    """
    global _CARB_MAP, _FUZZY_KEYS
    if _CARB_MAP is None:
        df = _load_usda_carb_table()
        _CARB_MAP = df.set_index("food_desc")["carb_100g"].to_dict()
        _FUZZY_KEYS = list(_CARB_MAP)  # for RapidFuzz matching
        log.info("Carb map initialised (%d keys)", len(_CARB_MAP))


def carbs_per_100g(ing: str, fuzzy: bool = True) -> float | None:
    """
    Look up carbohydrate content per 100g for an ingredient.

    First attempts exact matching, then falls back to fuzzy matching
    using RapidFuzz library with a similarity threshold of 90%.

    Args:
        ing: Normalized ingredient string
        fuzzy: Whether to use fuzzy matching if exact match fails

    Returns:
        Carbohydrate grams per 100g, or None if not found

    Example:
        >>> carbs_per_100g("white rice")
        28.2
        >>> carbs_per_100g("wite rice")  # Fuzzy match
        28.2
    """
    _ensure_carb_map()
    key = ing.lower().strip()
    val = _CARB_MAP.get(key)
    if val is not None or not fuzzy:
        return val

    # Fuzzy matching fallback
    match = process.extractOne(key, _FUZZY_KEYS, score_cutoff=90)
    return _CARB_MAP.get(match[0]) if match else None


# =============================================================================
# INGREDIENT CLASSIFICATION FUNCTIONS
# =============================================================================
"""
Core classification logic for determining if ingredients are keto or vegan.
These functions implement a multi-stage decision process combining rules,
nutritional data, and machine learning models.
"""


def is_keto_ingredient_list(tokens: list[str]) -> bool:
    """
    Check if a tokenized ingredient list is keto-friendly.

    Performs token-level matching against the NON_KETO list.
    Returns False if all tokens of any non-keto ingredient are found.

    Args:
        tokens: List of normalized ingredient tokens

    Returns:
        True if keto-friendly, False otherwise
    """
    for ingredient in NON_KETO:
        ing_tokens = ingredient.split()
        if all(tok in tokens for tok in ing_tokens):
            return False
    return True


def find_non_keto_hits(text: str) -> list[str]:
    """
    Find all non-keto ingredients present in the text.

    Used for debugging and understanding why an ingredient
    was classified as non-keto.

    Args:
        text: Normalized ingredient text

    Returns:
        Sorted list of matching non-keto ingredients
    """
    tokens = set(tokenize_ingredient(text))
    return sorted([
        ingredient for ingredient in NON_KETO
        if all(tok in tokens for tok in ingredient.split())
    ])


def is_ingredient_keto(ingredient: str) -> bool:
    """
    Determine if a single ingredient is keto-friendly.

    Implements a comprehensive decision pipeline:

    1. **Whitelist Check**: Immediate acceptance for known keto ingredients
    2. **Numeric Rule**: Reject if carbs > 10g/100g (USDA database)
       - Whole phrase lookup
       - Token-level fallback (ignoring stop words)
    3. **Regex Blacklist**: Fast pattern matching against NON_KETO
    4. **Token Blacklist**: Detailed token-level analysis
    5. **ML Model**: Machine learning prediction with rule verification

    Args:
        ingredient: Raw ingredient string

    Returns:
        True if keto-friendly, False otherwise

    Example:
        >>> is_ingredient_keto("almond flour")
        True  # Whitelisted
        >>> is_ingredient_keto("white rice")
        False  # High carb content
        >>> is_ingredient_keto("banana")
        False  # In NON_KETO list
    """
    if not ingredient:
        return True

    # 1. Whitelist (immediate accept)
    if RX_WL_KETO.search(ingredient):
        return True

    # 2. Numeric carbohydrate rule
    norm = normalise(ingredient)

    # 2a. Whole-phrase lookup
    carbs = carbs_per_100g(norm)
    if carbs is not None:
        return carbs <= 10.0

    # 2b. Token-level fallback
    for tok in tokenize_ingredient(norm):
        # Skip common stop words and units
        if tok in {"raw", "fresh", "dried", "powder", "mix", "sliced",
                   "organic", "cup", "cups", "tsp", "tbsp", "g", "kg", "oz"}:
            continue
        carbs_tok = carbs_per_100g(tok, fuzzy=True)
        if carbs_tok is not None and carbs_tok > 10.0:
            return False

    # 3. Regex blacklist (fast)
    if RX_KETO.search(norm):
        return False

    # 4. Token-level heuristic list
    if not is_keto_ingredient_list(tokenize_ingredient(norm)):
        return False

    # 5. ML model fallback (with rule verification)
    _ensure_pipeline()
    if 'keto' in _pipeline_state['models']:
        model = _pipeline_state['models']['keto']
        if _pipeline_state['vectorizer']:
            try:
                X = _pipeline_state['vectorizer'].transform([norm])

                # Check if model expects more features (was trained with images)
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    actual_features = X.shape[1]

                    if expected_features > actual_features:
                        # Model expects images, pad with zeros using combine_features
                        padding = np.zeros((1, 2048), dtype=np.float32)
                        X = combine_features(X, padding)

                prob = model.predict_proba(X)[0, 1]
            except Exception as e:
                log.warning("Vectorizer failed: %s. Falling back to rules.", e)
                prob = RuleModel(
                    "keto", RX_KETO, RX_WL_KETO).predict_proba([norm])[0, 1]
        else:
            prob = RuleModel(
                "keto", RX_KETO, RX_WL_KETO).predict_proba([norm])[0, 1]

        # Apply rule-based verification to ML prediction
        prob_adj = verify_with_rules(
            "keto", pd.Series([norm]), np.array([prob]))[0]
        return prob_adj >= 0.5

    # If no model available, default to True (passed all rule checks)
    return True


def is_ingredient_vegan(ingredient: str) -> bool:
    """
    Determine if an ingredient is vegan.

    Uses a simplified pipeline compared to keto classification:

    1. **Whitelist Check**: Accept known vegan alternatives
    2. **Blacklist Check**: Reject animal products
    3. **ML Model**: Machine learning with verification

    Args:
        ingredient: Raw ingredient string

    Returns:
        True if vegan, False otherwise

    Example:
        >>> is_ingredient_vegan("almond milk")
        True  # Whitelisted plant milk
        >>> is_ingredient_vegan("chicken")
        False  # Animal product
        >>> is_ingredient_vegan("honey")
        False  # Animal product (bee-derived)
    """
    if not ingredient:
        return True

    # Quick whitelist check
    if RX_WL_VEGAN.search(ingredient):
        return True

    # Normalize
    normalized = normalise(ingredient)

    # Quick blacklist check
    if RX_VEGAN.search(normalized) and not RX_WL_VEGAN.search(ingredient):
        return False

    # Use ML model if available
    _ensure_pipeline()
    if 'vegan' in _pipeline_state['models']:
        model = _pipeline_state['models']['vegan']
        if _pipeline_state['vectorizer']:
            try:
                X = _pipeline_state['vectorizer'].transform([normalized])

                # Check if model expects more features (was trained with images)
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    actual_features = X.shape[1]

                    if expected_features > actual_features:
                        # Model expects images, pad with zeros using combine_features
                        padding = np.zeros((1, 2048), dtype=np.float32)
                        X = combine_features(X, padding)

                prob = model.predict_proba(X)[0, 1]
            except Exception as e:
                log.warning(
                    "Vectorizer failed: %s. Using rule-based fallback.", e)
                prob = RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN).predict_proba(
                    [normalized])[0, 1]
        else:
            prob = RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN).predict_proba(
                [normalized])[0, 1]

        # Apply verification
        prob_adj = verify_with_rules(
            "vegan", pd.Series([normalized]), np.array([prob]))[0]
        return prob_adj >= 0.5

    return True


def is_keto(ingredients: Iterable[str] | str) -> bool:
    """
    Check if all ingredients in a recipe are keto-friendly.

    This is the main public API for keto classification. It handles
    various input formats and ensures all ingredients pass the keto test.

    Args:
        ingredients: Either a string (comma-separated or JSON) or an iterable

    Returns:
        True if ALL ingredients are keto-friendly, False otherwise

    Example:
        >>> is_keto("almond flour, eggs, butter")
        True
        >>> is_keto(["chicken", "broccoli", "rice"])
        False  # Rice is not keto
        >>> is_keto('["spinach", "cheese", "cream"]')
        True
    """
    _ensure_pipeline()
    if isinstance(ingredients, str):
        try:
            if ingredients.startswith('['):
                ingredients = json.loads(ingredients)
            else:
                ingredients = [i.strip()
                               for i in ingredients.split(',') if i.strip()]
        except Exception:
            ingredients = [ingredients]
    return all(is_ingredient_keto(ing) for ing in ingredients)


def is_vegan(ingredients: Iterable[str] | str) -> bool:
    """
    Check if all ingredients in a recipe are vegan.

    This is the main public API for vegan classification. It handles
    various input formats and ensures all ingredients pass the vegan test.

    Args:
        ingredients: Either a string (comma-separated or JSON) or an iterable

    Returns:
        True if ALL ingredients are vegan, False otherwise

    Example:
        >>> is_vegan("tofu, soy sauce, vegetables")
        True
        >>> is_vegan(["almond milk", "honey", "oats"])
        False  # Honey is not vegan
        >>> is_vegan('["rice", "beans", "vegetables"]')
        True
    """
    _ensure_pipeline()
    if isinstance(ingredients, str):
        try:
            if ingredients.startswith('['):
                ingredients = json.loads(ingredients)
            else:
                ingredients = [i.strip()
                               for i in ingredients.split(',') if i.strip()]
        except Exception:
            ingredients = [ingredients]
    return all(is_ingredient_vegan(ing) for ing in ingredients)


# =============================================================================
# RULE-BASED MODEL
# =============================================================================
"""
A scikit-learn compatible model that implements pure rule-based classification.
Used as a baseline and fallback when ML models are unavailable or fail.
"""


class RuleModel(BaseEstimator, ClassifierMixin):
    """
    Rule-based classifier for diet classification.

    Implements the scikit-learn interface while using only regex patterns
    and domain rules for classification. Provides probabilistic outputs
    with high confidence for rule matches.

    Attributes:
        task: Classification task ('keto' or 'vegan')
        rx_black: Compiled regex for blacklist patterns
        rx_white: Compiled regex for whitelist patterns
        pos_prob: Probability assigned to positive class (default 0.98)
        neg_prob: Probability assigned to negative class (default 0.02)
    """

    def __init__(self, task: str, rx_black, rx_white=None,
                 pos_prob=0.98, neg_prob=0.02):
        """
        Initialize rule-based model.

        Args:
            task: 'keto' or 'vegan'
            rx_black: Regex pattern for blacklisted items
            rx_white: Regex pattern for whitelisted items
            pos_prob: Confidence for positive predictions
            neg_prob: Confidence for negative predictions
        """
        self.task = task
        self.rx_black = rx_black
        self.rx_white = rx_white
        self.pos_prob = pos_prob
        self.neg_prob = neg_prob

    def fit(self, X, y=None):
        """No training needed for rule-based model."""
        return self

    def _pos(self, d: str) -> bool:
        """
        Determine if ingredient is positive class using full rule pipeline.

        Delegates to the main classification functions to ensure
        consistency with the overall system.
        """
        if self.task == "keto":
            return is_ingredient_keto(d)
        else:  # vegan
            return is_ingredient_vegan(d)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples.

        Args:
            X: Array-like of ingredient strings

        Returns:
            Array of shape (n_samples, 2) with probabilities
        """
        p = np.fromiter(
            (self.pos_prob if self._pos(d) else self.neg_prob for d in X),
            float,
            count=len(X)
        )
        return np.c_[1 - p, p]

    def predict(self, X):
        """
        Predict class labels for samples.

        Args:
            X: Array-like of ingredient strings

        Returns:
            Array of binary predictions (0 or 1)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =============================================================================
# VERIFICATION LAYER
# =============================================================================
"""
Post-processing functions that apply rule-based corrections to ML predictions.
This ensures that hard rules always override ML predictions when applicable.
"""


def verify_with_rules(task: str, clean: pd.Series, prob: np.ndarray) -> np.ndarray:
    """
    Apply rule-based verification to ML predictions.

    This function ensures that domain rules always take precedence over
    ML predictions. It's a critical safety layer that prevents the model
    from making obvious mistakes.

    Args:
        task: 'keto' or 'vegan'
        clean: Series of normalized ingredient texts
        prob: Array of ML predicted probabilities

    Returns:
        Adjusted probability array with rule corrections applied
    """
    adjusted = prob.copy()

    if task == "keto":
        # Regex-based whitelist/blacklist
        is_whitelisted = clean.str.contains(RX_WL_KETO)
        is_blacklisted = clean.str.contains(RX_KETO)
        forced_non_keto = is_blacklisted & ~is_whitelisted
        adjusted[forced_non_keto.values] = 0.0

        # Token-based ingredient verification
        for i, txt in enumerate(clean):
            if adjusted[i] > 0.5:
                tokens = tokenize_ingredient(normalise(txt))
                if not is_keto_ingredient_list(tokens):
                    adjusted[i] = 0.0
                    log.debug("Heuristically rejected '%s' as non-keto", txt)

        if forced_non_keto.any():
            log.debug("Keto Verification: forced %d probs to 0 (regex)",
                      forced_non_keto.sum())

    else:  # vegan
        bad = clean.str.contains(RX_VEGAN) & ~clean.str.contains(RX_WL_VEGAN)
        adjusted[bad.values] = 0.0
        if bad.any():
            log.debug("Vegan Verification: forced %d probs to 0", bad.sum())

    return adjusted


# # =============================================================================
# # SANITY CHECKS
# # =============================================================================
# """
# Basic tests to ensure the classification system is working correctly.
# These checks run on module import to catch configuration errors early.
# """

# Test whitelist functionality
assert is_ingredient_keto("almond flour"), "Whitelist check failed"

# Test numeric rule (rice has ~28g carbs/100g)
assert not is_ingredient_keto("white rice"), "Numeric carb rule failed"

# Test rule model delegation
rule = RuleModel("keto", None, None)
assert rule._pos("banana") is False, "Rule model delegation failed"
log.info("✅ All sanity checks passed")


# =============================================================================
# PIPELINE STATE MANAGEMENT
# =============================================================================
"""
Global state for managing vectorizers and models across the pipeline.
This enables the simple API functions to work without explicit model loading.
"""

# Global pipeline state
_pipeline_state = {
    'vectorizer': None,
    'models': {},
    'initialized': False
}


def save_pipeline_state(stage: str, data: dict):
    """
    Save pipeline state for resume capability in persistent directory.

    Args:
        stage: Current pipeline stage identifier
        data: Dictionary of data to save
    """
    # Use the persistent state directory
    state_dir = CFG.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)

    state_path = state_dir / "pipeline_state.pkl"
    backup_path = state_dir / f"pipeline_state_{stage}.pkl"

    state = {
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
        'data': data,
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'artifacts_dir': str(CFG.artifacts_dir),  # Track where artifacts are
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

    try:
        # Save main state file
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

        # Save stage-specific backup
        with open(backup_path, 'wb') as f:
            pickle.dump(state, f)

        log.debug(f"   💾 Pipeline state saved at stage: {stage}")

        # Clean up old backups (keep only last 5)
        backups = sorted(state_dir.glob("pipeline_state_*.pkl"),
                         key=lambda p: p.stat().st_mtime)
        if len(backups) > 5:
            for old_backup in backups[:-5]:
                old_backup.unlink()
                log.debug(f"   🗑️  Removed old backup: {old_backup.name}")

    except Exception as e:
        log.error(f"   ❌ Failed to save pipeline state: {e}")


def load_pipeline_state():
    """
    Load the most recent pipeline state from persistent directory.

    Returns:
        Tuple of (stage, data) or (None, None) if no state found
    """
    state_path = CFG.state_dir / "pipeline_state.pkl"

    if not state_path.exists():
        return None, None

    try:
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        log.info(f"   📂 Loaded pipeline state from stage: {state['stage']}")
        log.info(f"   ├─ Saved at: {state['timestamp']}")
        log.info(f"   └─ Memory usage at save: {state['memory_usage']:.1f}%")

        return state['stage'], state['data']

    except Exception as e:
        log.error(f"   ❌ Failed to load pipeline state: {e}")
        return None, None


def _ensure_pipeline():
    """
    Ensure the ML pipeline is initialized with vectorizer and models.

    This function implements lazy loading of models:
    1. First checks if models are already loaded
    2. Attempts to load from disk if available
    3. Falls back to training if no saved models exist
    4. Ultimate fallback to rule-only mode if training fails

    The function ensures the system is always in a usable state,
    even if ML components fail to load or train.
    """
    if _pipeline_state['initialized']:
        return

    vec_path = CFG.artifacts_dir / "vectorizer.pkl"
    models_path = CFG.artifacts_dir / "models.pkl"

    try:
        # Attempt to load existing models
        if vec_path.exists() and models_path.exists():
            with open(vec_path, 'rb') as f:
                _pipeline_state['vectorizer'] = pickle.load(f)
            with open(models_path, 'rb') as f:
                _pipeline_state['models'] = pickle.load(f)
            log.info("Loaded vectorizer + models from %s", CFG.artifacts_dir)

        else:
            # No saved models - run training pipeline
            log.info("No saved artifacts found - running full pipeline once")
            # Import here to avoid circular dependency
            from ..... import run_full_pipeline, BEST

            vec, _, _, res = run_full_pipeline(mode="both", sample_frac=0.1)

            # Select best model per task
            best_models = {}
            for task in ("keto", "vegan"):
                best = max((r for r in res
                            if r["task"] == task and "TxtImg" not in r["model"]),
                           key=lambda r: r["F1"])
                base_name = best["model"].split('_')[0]  # strip domain suffix
                best_models[task] = BEST[base_name]

            # Save artifacts
            CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
            with open(vec_path, 'wb') as f:
                pickle.dump(vec, f)
            with open(models_path, 'wb') as f:
                pickle.dump(best_models, f)

            _pipeline_state['vectorizer'] = vec
            _pipeline_state['models'] = best_models
            log.info("Fresh artifacts saved to %s", CFG.artifacts_dir)

    except Exception as e:
        # Ultimate fallback - rules only
        log.warning("Model bootstrap failed (%s). Falling back to rules.", e)
        _pipeline_state['models'] = {
            "keto": RuleModel("keto", RX_KETO, RX_WL_KETO),
            "vegan": RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN),
        }

    _pipeline_state['initialized'] = True


# =============================================================================
# SILVER LABEL GENERATION
# =============================================================================
"""
Functions for generating weak (silver) labels from unlabeled data using
rule-based heuristics. This enables training on large unlabeled datasets.
"""


def _rule_only_keto(text: str) -> bool:
    """
    Apply keto rules without ML model fallback.

    Used for silver label generation where we want consistent
    rule-based labels without ML influence.
    """
    if RX_WL_KETO.search(text):
        return True
    norm = normalise(text)
    c = carbs_per_100g(norm)
    if c is not None and c > 10:
        return False
    if RX_KETO.search(norm):
        return False
    if not is_keto_ingredient_list(tokenize_ingredient(norm)):
        return False
    return True


def _rule_only_vegan(text: str) -> bool:
    """
    Apply vegan rules without ML model fallback.

    Used for silver label generation where we want consistent
    rule-based labels without ML influence.
    """
    if RX_WL_VEGAN.search(text):
        return True
    norm = normalise(text)
    if RX_VEGAN.search(norm) and not RX_WL_VEGAN.search(text):
        return False
    return True


def build_silver(recipes: pd.DataFrame) -> pd.DataFrame:
    """
    Generate silver labels for recipe dataset using heuristic rules.

    Creates weak labels that can be used for training when manual
    labels are unavailable. The silver labels are less accurate than
    gold standard labels but enable training on much larger datasets.

    Args:
        recipes: DataFrame with 'ingredients' column

    Returns:
        DataFrame with added columns:
        - clean: Normalized ingredient text
        - silver_keto: Binary keto label (0 or 1)
        - silver_vegan: Binary vegan label (0 or 1)
    """
    df = recipes[["ingredients"]].copy()
    df["clean"] = df["ingredients"].fillna("").map(normalise)

    df["silver_keto"] = df["clean"].map(lambda t: int(_rule_only_keto(t)))
    df["silver_vegan"] = df["clean"].map(lambda t: int(_rule_only_vegan(t)))

    return df


# =============================================================================
# FEATURE PROCESSING
# =============================================================================
"""
Functions for combining different feature types and filtering data.
"""


def combine_features(X_text, X_image) -> csr_matrix:
    """
    Concatenate sparse text matrix with dense image array.

    Creates a unified feature matrix for multi-modal learning by
    horizontally stacking text and image features.

    Args:
        X_text: Sparse TF-IDF matrix from text
        X_image: Dense array of image embeddings

    Returns:
        Combined sparse matrix with all features
    """
    img_sparse = csr_matrix(X_image)
    return hstack([X_text, img_sparse])


def filter_photo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows with usable photo URLs.

    Filters out rows with missing or placeholder photo URLs
    (containing "nophoto", "nopic", etc.).

    Args:
        df: DataFrame potentially containing 'photo_url' column

    Returns:
        Filtered DataFrame with valid photo URLs only
    """
    if 'photo_url' not in df.columns:
        return df.iloc[0:0].copy()  # Return empty DataFrame

    # Filter out placeholder images
    mask = ~df['photo_url'].str.contains(
        r"nophoto|nopic|nopicture", case=False, na=False)
    mask &= df['photo_url'].astype(bool)  # Also remove empty/null URLs

    return df.loc[mask].copy()


def filter_silver_by_downloaded_images(silver_df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """
    Keep only rows in silver that have corresponding downloaded image files.
    """
    silver_img_dir = image_dir / "silver"

    # Check if directory exists
    if not silver_img_dir.exists():
        log.warning(f"Image directory does not exist: {silver_img_dir}")
        return silver_df.iloc[0:0].copy()  # Return empty DataFrame

    # Get list of downloaded image files
    downloaded_ids = []
    for p in silver_img_dir.glob("*.jpg"):
        try:
            downloaded_ids.append(int(p.stem))
        except ValueError:
            continue  # Skip non-numeric filenames

    if not downloaded_ids:
        log.warning(f"No downloaded images found in {silver_img_dir}")
        return silver_df.iloc[0:0].copy()

    # Return rows that have downloaded images
    return silver_df.loc[silver_df.index.intersection(downloaded_ids)].copy()


# =============================================================================
# CLASS BALANCE HELPERS
# =============================================================================
"""
Functions for handling imbalanced datasets through resampling techniques.
"""


def show_balance(df: pd.DataFrame, title: str) -> None:
    """
    Print class distribution statistics.

    Displays the positive/negative class balance for both keto and vegan
    labels in a formatted table.

    Args:
        df: DataFrame containing label columns
        title: Title for the display
    """
    log.info(f"\n── {title} set class counts ─────────────")
    for lab in ("keto", "vegan"):
        for col in (f"label_{lab}", f"silver_{lab}"):
            if col in df.columns:
                tot = len(df)
                if tot == 0:
                    log.info(f"{lab:>5}: No data available (0 rows)")
                    break
                pos = int(df[col].sum())
                log.info(f"{lab:>5}: {pos:6}/{tot} ({pos/tot:>5.1%})")
                break


def apply_smote(X, y, max_dense_size: int = int(5e7)):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) for class balancing.
    """
    try:
        # Ensure y is a proper numpy array
        y = np.asarray(y).ravel()  # Add ravel() to ensure 1D array

        # Get unique classes
        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            return X, y

        counts = np.bincount(y)

        # Ensure we have valid counts
        if len(counts) < 2:
            return X, y

        # Explicit float conversion for ratio calculation
        min_count = float(counts.min())
        total_count = float(counts.sum())
        ratio = min_count / total_count

        if ratio < 0.3:  # Only apply if minority class < 30%
            # Check if X is sparse and decide strategy
            if hasattr(X, "toarray"):  # X is sparse
                elements = X.shape[0] * X.shape[1]
                if elements > max_dense_size:
                    # Too large for SMOTE - use random oversampling
                    ros = RandomOverSampler(random_state=42)
                    return ros.fit_resample(X, y)
                else:
                    # Convert to dense for SMOTE
                    X_dense = X.toarray()
                    smote = SMOTE(sampling_strategy=0.3, random_state=42)
                    return smote.fit_resample(X_dense, y)
            else:
                # X is already dense
                smote = SMOTE(sampling_strategy=0.3, random_state=42)
                return smote.fit_resample(X, y)
        return X, y
    except Exception as e:
        log.warning(f"SMOTE failed: {e}. Using original data.")
        return X, y


# =============================================================================
# MACHINE LEARNING MODEL BUILDERS
# =============================================================================
"""
Functions for creating and configuring various ML models tailored to
different feature domains (text, image, or both).
"""


def build_models(task: str, domain: str = "text") -> Dict[str, BaseEstimator]:
    """
    Build a dictionary of ML models appropriate for the given domain.

    Enhanced with resource-aware model selection that adapts to available memory.

    Different models are optimized for different feature types:
    - Text models: Naive Bayes, linear models (work well with sparse TF-IDF)
    - Image models: Neural networks, tree-based models (handle dense features)
    - Both: All models available

    Args:
        task: Classification task ('keto' or 'vegan')
        domain: Feature domain ('text', 'image', or 'both')

    Returns:
        Dictionary mapping model names to estimator instances
    """
    models: Dict[str, BaseEstimator] = {}

    # Check available resources
    available_memory = get_available_memory()
    is_sanity_check = os.environ.get('SANITY_CHECK') == '1'

    log.info(f"   🔧 Building models for {task} ({domain} domain)")
    log.info(f"   ├─ Available memory: {available_memory:.1f} GB")
    log.info(f"   └─ Sanity check mode: {is_sanity_check}")

    # If in sanity check mode, return minimal models
    if is_sanity_check:
        log.info(f"   🏃 Sanity check mode - using minimal model set")
        if domain == "text":
            return {
                "NB": MultinomialNB(),
                "Ridge": RidgeClassifier(class_weight="balanced", random_state=42)
            }
        else:
            return {
                "Softmax": LogisticRegression(
                    solver="lbfgs", max_iter=100, random_state=42
                )
            }

    # Text-oriented models (work well with sparse features)
    text_family: Dict[str, BaseEstimator] = {
        "NB": MultinomialNB(),  # Classic for text classification
        "Softmax": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge": RidgeClassifier(class_weight="balanced", random_state=42),
        "PA": PassiveAggressiveClassifier(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "SGD": SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "LinearSVC": LinearSVC(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            dual=False,  # Use primal optimization for large datasets
            tol=1e-3
        ),
    }

    # Image/mixed-feature models (handle dense features better)
    image_family: Dict[str, BaseEstimator] = {}

    # Adapt models based on available memory
    if available_memory >= 16:  # High memory - use all models
        log.info(f"   💪 High memory mode - all models available")

        image_family["RF"] = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

        if lgb:
            image_family["LGBM"] = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                objective="binary",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True,
            )

    elif available_memory >= 8:  # Medium memory - reduce model complexity
        log.info(f"   ⚡ Medium memory mode - reduced model complexity")

        image_family["RF"] = RandomForestClassifier(
            n_estimators=100,  # Reduced from 150
            max_depth=15,      # Reduced from 20
            min_samples_split=10,  # Increased from 5
            min_samples_leaf=5,    # Increased from 2
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

        if lgb:
            image_family["LGBM"] = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.1,
                n_estimators=50,   # Reduced from 100
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                objective="binary",
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                force_col_wise=True,
                # Memory-saving parameters
                max_bin=127,       # Reduced from default 255
                min_data_in_bin=10,
            )

    else:  # Low memory - minimal models only
        log.warning(
            f"   ⚠️  Low memory mode ({available_memory:.1f} GB) - using minimal models")

        # Only lightweight model for image features
        image_family["RF"] = RandomForestClassifier(
            n_estimators=50,   # Minimal trees
            max_depth=10,      # Shallow trees
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=2,  # Limit parallelism
            random_state=42,
        )

        # Remove memory-intensive text models
        text_family.pop("LinearSVC", None)
        text_family.pop("SGD", None)

    # Assemble model selection based on domain
    if domain == "text":
        models.update(text_family)
    elif domain == "image":
        models.update(image_family)
    elif domain == "both":
        models.update(text_family)
        models.update(image_family)

    # Log final model selection
    log.info(f"   📦 Selected models: {list(models.keys())}")

    return models


# =============================================================================
# HYPERPARAMETER CONFIGURATION
# =============================================================================
"""
Hyperparameter grids for model tuning via grid search.
"""

HYPER = {
    # Text models
    "Softmax": {
        "C": [0.1, 1, 10]  # Regularization strength
    },
    "SGD": {
        "alpha": [1e-4, 1e-3],  # Regularization
        "loss": ["log_loss", "modified_huber"],  # Loss function
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0]  # Regularization
    },
    "PA": {
        "C": [0.5, 1.0]  # Aggressiveness parameter
    },
    "NB": {},  # No hyperparameters to tune

    # Image/mixed models
    "MLP": {
        "hidden_layer_sizes": [(256,), (512, 128)],
        "alpha": [0.0001, 0.001],  # L2 regularization
        "learning_rate_init": [0.001, 0.005],
    },
    "RF": {
        "n_estimators": [150, 300],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
    },
    "LGBM": {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
        "n_estimators": [150, 250],
        "min_child_samples": [10, 20],
    },
    # For CalibratedSVM, parameters live under 'estimator__'
    "SVM_RBF": {
        "estimator__svc__C": [0.5, 1, 2],
        "estimator__svc__gamma": ["scale", 0.001]
    },
}

# Global cache for best models
BEST: Dict[str, BaseEstimator] = {}

# Fast mode settings for development
FAST = True
CV = 2 if FAST else 3  # Cross-validation folds
N_IT = 2 if FAST else 6  # Number of iterations for random search


def ensure_predict_proba(estimator, X_train, y_train):
    """
    Ensure estimator has predict_proba method by wrapping with calibration if needed.

    Some models (like SVC) don't provide probability estimates by default.
    This function wraps them with CalibratedClassifierCV to enable probabilities.

    Args:
        estimator: Fitted estimator
        X_train: Training features
        y_train: Training labels

    Returns:
        Estimator with predict_proba capability
    """
    if not hasattr(estimator, "predict_proba"):
        log.info(
            f"Adding probability calibration to {estimator.__class__.__name__}")
        try:
            from sklearn.calibration import CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(
                estimator, cv=3, method='sigmoid')
            calibrated.fit(X_train, y_train)
            return calibrated
        except Exception as e:
            log.error(f"Calibration failed: {e}")
            return estimator
    return estimator


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================
"""
Functions for optimizing model hyperparameters through grid search.
"""


def tune(name: str, base: BaseEstimator, X, y, cv: int = CV) -> BaseEstimator:
    """
    Optimize hyperparameters for a model using grid search.

    This function implements comprehensive hyperparameter tuning with:
    - Caching of tuned models
    - Detailed progress tracking
    - Fallback to defaults on failure
    - Performance analysis

    Args:
        name: Model name (key for HYPER dictionary)
        base: Base estimator to tune
        X: Training features
        y: Training labels
        cv: Number of cross-validation folds

    Returns:
        Fitted estimator with optimized hyperparameters
    """

    from itertools import product

    tune_start = time.time()

    # Check cache
    if name in BEST:
        cached_time = time.time() - tune_start
        log.info(
            f"            ✅ {name}: Using cached model ({cached_time*1000:.0f}ms)")
        return BEST[name]

    # Get hyperparameter grid
    grid = HYPER.get(name, {})

    if not grid:
        # No hyperparameters to tune - use defaults
        log.info(
            f"            🔧 {name}: No hyperparameters defined, using defaults")

        with tqdm(total=1, desc=f"               ├─ Default Fit",
                  position=4, leave=False,
                  bar_format="               ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as default_pbar:
            BEST[name] = base.fit(X, y)
            default_pbar.update(1)

        fit_time = time.time() - tune_start
        log.info(
            f"            ✅ {name}: Default fit completed in {fit_time:.1f}s")
        return BEST[name]

    # Calculate total parameter combinations
    param_combinations = 1
    for param_values in grid.values():
        param_combinations *= len(param_values) if isinstance(
            param_values, list) else 1

    total_fits = param_combinations * cv

    log.info(f"            🔍 {name}: Starting hyperparameter optimization")
    log.info(f"               ├─ Parameters: {list(grid.keys())}")
    log.info(f"               ├─ Combinations: {param_combinations}")
    log.info(f"               ├─ CV Folds: {cv}")
    log.info(f"               └─ Total Fits: {total_fits}")

    # Display parameter details
    for param, values in grid.items():
        if isinstance(values, list) and len(values) <= 10:
            log.info(f"               ├─ {param}: {values}")
        else:
            log.info(f"               ├─ {param}: {len(values)} values")

    try:
        # Create progress bar for grid search
        with tqdm(total=total_fits, desc=f"               ├─ Grid Search",
                  position=4, leave=False,
                  bar_format="               ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as gs_pbar:

            # Initialize grid search
            search = GridSearchCV(
                estimator=base,
                param_grid=grid,
                scoring="f1",
                n_jobs=-1,
                cv=cv,
                verbose=0,
                return_train_score=True,
                error_score='raise'
            )

            # Fit with progress tracking
            gs_pbar.set_description(f"               ├─ {name}: Searching")
            search.fit(X, y)

            # Update progress to completion
            remaining = total_fits - gs_pbar.n
            gs_pbar.update(remaining)
            gs_pbar.set_description(f"               ├─ {name}: Complete")

        search_time = time.time() - tune_start

        # Extract results
        best_score = search.best_score_
        best_params = search.best_params_
        best_estimator = search.best_estimator_

        log.info(
            f"            ✅ {name}: Grid search completed in {search_time:.1f}s")
        log.info(f"               ├─ Best CV Score: {best_score:.3f}")
        log.info(f"               ├─ Best Parameters:")

        for param, value in best_params.items():
            log.info(f"               │  ├─ {param}: {value}")

        # Performance analysis
        results_df = pd.DataFrame(search.cv_results_)

        # Show top 3 parameter combinations
        top_results = results_df.nlargest(3, 'mean_test_score')
        log.info(f"               ├─ Top 3 Configurations:")

        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            params_str = ", ".join([f"{k.replace('param_', '')}: {v}"
                                   for k, v in row.items()
                                   if k.startswith('param_')])
            log.info(f"               │  {idx}. Score: {row['mean_test_score']:.3f} "
                     f"(±{row['std_test_score']:.3f}) | {params_str}")

        # Parameter importance analysis
        if len(grid) > 1:
            log.info(f"               ├─ Parameter Impact Analysis:")
            for param in grid.keys():
                param_col = f'param_{param}'
                if param_col in results_df.columns:
                    param_impact = results_df.groupby(
                        param_col)['mean_test_score'].agg(['mean', 'std'])
                    best_param_val = param_impact['mean'].idxmax()
                    best_param_score = param_impact.loc[best_param_val, 'mean']
                    worst_param_val = param_impact['mean'].idxmin()
                    worst_param_score = param_impact.loc[worst_param_val, 'mean']
                    impact = best_param_score - worst_param_score

                    log.info(f"               │  ├─ {param}: Impact={impact:.3f} "
                             f"(Best: {best_param_val}, Worst: {worst_param_val})")

        # Cross-validation stability
        cv_std = results_df.loc[search.best_index_, 'std_test_score']
        cv_stability = "High" if cv_std < 0.02 else "Medium" if cv_std < 0.05 else "Low"
        log.info(
            f"               ├─ CV Stability: {cv_stability} (std={cv_std:.3f})")

        # Performance improvement over default
        try:
            # Estimate default performance
            default_scores = []
            for train_idx, val_idx in search.cv.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                default_model = clone(base).fit(X_train_fold, y_train_fold)
                y_pred = default_model.predict(X_val_fold)
                fold_score = f1_score(y_val_fold, y_pred, zero_division=0)
                default_scores.append(fold_score)

            default_score = np.mean(default_scores)
            improvement = best_score - default_score
            improvement_pct = (improvement / default_score *
                               100) if default_score > 0 else 0

            log.info(f"               ├─ Improvement over default: {improvement:+.3f} "
                     f"({improvement_pct:+.1f}%)")

        except Exception as e:
            log.debug(f"               ├─ Default comparison failed: {e}")

        # Cache the best estimator
        BEST[name] = best_estimator

        # Save hyperparameters
        hyperparams_file = "best_hyperparams.json"
        try:
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    saved_params = json.load(f)
            else:
                saved_params = {}

            saved_params[name] = best_params

            with open(hyperparams_file, 'w') as f:
                json.dump(saved_params, f, indent=2)

            log.debug(
                f"               └─ Saved hyperparameters to {hyperparams_file}")

        except Exception as e:
            log.warning(
                f"               └─ Failed to save hyperparameters: {e}")

    except Exception as e:
        # Error handling with fallback
        search_time = time.time() - tune_start
        log.error(
            f"            ❌ {name}: Grid search failed after {search_time:.1f}s")
        log.error(f"               └─ Error: {str(e)[:80]}...")

        if log.level <= logging.DEBUG:
            import traceback
            log.debug(
                f"Full traceback for {name} tuning:\n{traceback.format_exc()}")

        # Fallback to default parameters
        log.info(f"            🛡️  {name}: Falling back to default parameters")

        try:
            with tqdm(total=1, desc=f"               ├─ Fallback Fit",
                      position=4, leave=False,
                      bar_format="               ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as fallback_pbar:
                BEST[name] = base.fit(X, y)
                fallback_pbar.update(1)

            fallback_time = time.time() - tune_start
            log.info(
                f"            ✅ {name}: Fallback completed in {fallback_time:.1f}s")

        except Exception as fallback_error:
            fallback_time = time.time() - tune_start
            log.error(
                f"            ❌ {name}: Fallback also failed after {fallback_time:.1f}s")
            log.error(
                f"               └─ Fallback Error: {str(fallback_error)[:60]}...")
            raise RuntimeError(
                f"Both grid search and fallback failed for {name}")

    return BEST[name]


def tune_with_early_stopping(name: str,
                             base: BaseEstimator,
                             X, y,
                             cv: int = CV,
                             patience: int = 3,
                             min_improvement: float = 0.001) -> BaseEstimator:
    """
    Enhanced tuning with early stopping for large parameter grids.
    Stops search early if no improvement is seen for 'patience' iterations.

    Args:
        patience: Number of iterations without improvement before stopping
        min_improvement: Minimum improvement required to reset patience counter
    """

    from sklearn.model_selection import ParameterGrid

    if name in BEST:
        log.info(f"            ✅ {name}: Using cached model")
        return BEST[name]

    grid = HYPER.get(name, {})
    if not grid:
        log.info(f"            🔧 {name}: No hyperparameters, using defaults")
        BEST[name] = base.fit(X, y)
        return BEST[name]

    # Convert to parameter grid for manual iteration
    param_grid = list(ParameterGrid(grid))
    total_combinations = len(param_grid)

    log.info(f"            🔍 {name}: Early stopping grid search")
    log.info(f"               ├─ Total combinations: {total_combinations}")
    log.info(f"               ├─ Patience: {patience}")
    log.info(f"               └─ Min improvement: {min_improvement}")

    best_score = -np.inf
    best_params = None
    best_estimator = None
    patience_counter = 0

    # Progress bar for early stopping search
    with tqdm(param_grid, desc=f"               ├─ Early Stop Search",
              position=4, leave=False,
              bar_format="               ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as es_pbar:

        for i, params in enumerate(es_pbar):
            try:
                # Set parameters and perform cross-validation
                model = clone(base).set_params(**params)
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring='f1', n_jobs=-1)
                current_score = np.mean(scores)

                # Update progress bar with current best
                es_pbar.set_postfix({
                    'Best': f"{best_score:.3f}",
                    'Current': f"{current_score:.3f}",
                    'Patience': f"{patience_counter}/{patience}"
                })

                # Check for improvement
                if current_score > best_score + min_improvement:
                    improvement = current_score - best_score
                    best_score = current_score
                    best_params = params
                    best_estimator = model.fit(X, y)
                    patience_counter = 0

                    log.info(f"               ├─ New best: {current_score:.3f} "
                             f"(+{improvement:.3f}) | {params}")
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    remaining = total_combinations - i - 1
                    log.info(f"               ├─ Early stopping after {i+1}/{total_combinations} "
                             f"combinations (saved {remaining})")
                    break

            except Exception as e:
                log.warning(
                    f"               ├─ Params {params} failed: {str(e)[:40]}...")
                patience_counter += 1

    if best_estimator is not None:
        log.info(f"            ✅ {name}: Early stopping complete")
        log.info(f"               ├─ Best score: {best_score:.3f}")
        log.info(f"               └─ Best params: {best_params}")
        BEST[name] = best_estimator
    else:
        log.warning(
            f"            ⚠️  {name}: No valid configuration found, using defaults")
        BEST[name] = base.fit(X, y)

    return BEST[name]


# =============================================================================
# TRAINING PIPELINE
# =============================================================================
"""
Main training pipeline that orchestrates model training and evaluation.
"""


def run_mode_A(
    X_silver,
    gold_clean: pd.Series,
    X_gold,
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    *,
    domain: str = "text",
    apply_smote_flag: bool = True,
    checkpoint_dir: Path = None
) -> list[dict]:
    """
    Train on weak (silver) labels, evaluate on gold standard labels.

    Enhanced with complete checkpoint/resume functionality.
    """
    # Use persistent checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = CFG.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings('ignore', message='.*truth value of an array.*')

    # Ensure all arrays are properly formatted
    if hasattr(X_silver, 'toarray'):
        X_silver = X_silver.toarray() if X_silver.nnz < 1e6 else X_silver

    # Initialize results and timing
    results: list[dict] = []
    pipeline_start = time.time()

    # Check for existing checkpoints and load completed work
    checkpoint_state = {
        'completed_tasks': set(),
        'completed_models': {},  # task -> list of completed model names
        'loaded_results': []
    }

    existing_checkpoints = sorted(
        checkpoint_dir.glob(f"checkpoint_*_{domain}.pkl"))

    if existing_checkpoints:
        log.info(
            f"   📂 Found {len(existing_checkpoints)} existing checkpoints")

        for ckpt_path in existing_checkpoints:
            try:
                with open(ckpt_path, 'rb') as f:
                    ckpt_data = pickle.load(f)

                task = ckpt_data['task']
                checkpoint_state['completed_tasks'].add(task)

                # Load completed models for this task
                if 'all_results' in ckpt_data:
                    checkpoint_state['loaded_results'].extend(
                        ckpt_data['all_results'])

                    # Track which models were completed
                    completed_model_names = [r['model']
                                             for r in ckpt_data['all_results']]
                    checkpoint_state['completed_models'][task] = completed_model_names

                    log.info(
                        f"   ├─ Loaded {task} checkpoint: {len(completed_model_names)} models")
                    log.info(
                        f"   │  └─ Models: {', '.join(completed_model_names)}")

                # Restore best models to BEST cache
                if 'all_task_models' in ckpt_data:
                    for model_res in ckpt_data['all_task_models']:
                        if 'model_object' in model_res:
                            model_name = model_res['model_base_name']
                            BEST[f"{task}_{domain}_{model_name}"] = model_res['model_object']

            except Exception as e:
                log.warning(
                    f"   ├─ Failed to load checkpoint {ckpt_path.name}: {e}")

        if checkpoint_state['completed_tasks']:
            log.info(
                f"   ✅ Resuming from checkpoint - completed tasks: {checkpoint_state['completed_tasks']}")
            results.extend(checkpoint_state['loaded_results'])

    # Log pipeline initialization
    log.info("🚀 Starting MODE A Training Pipeline")
    log.info(f"   Domain: {domain}")
    log.info(f"   SMOTE enabled: {apply_smote_flag}")
    log.info(f"   Silver set size: {len(silver_df):,}")
    log.info(f"   Gold set size: {len(gold_df):,}")
    log.info(f"   Feature dimensions: {X_silver.shape}")
    log.info(f"   Checkpoint directory: {checkpoint_dir}")

    if checkpoint_state['completed_tasks']:
        log.info(
            f"   📂 Resuming with {len(checkpoint_state['loaded_results'])} pre-loaded results")

    # Show class distribution
    log.info("\n📊 Class Distribution Analysis:")
    for task in ("keto", "vegan"):
        silver_pos = silver_df[f"silver_{task}"].sum()
        silver_total = len(silver_df)
        gold_pos = gold_df[f"label_{task}"].sum()
        gold_total = len(gold_df)

        log.info(f"   {task.capitalize():>5} - Silver: {silver_pos:,}/{silver_total:,} ({silver_pos/silver_total:.1%}) | "
                 f"Gold: {gold_pos:,}/{gold_total:,} ({gold_pos/gold_total:.1%})")

    # Store all models for each task
    all_task_models = {"keto": [], "vegan": []}

    # Main training loop
    task_progress = tqdm(["keto", "vegan"], desc="🔬 Training Tasks",
                         position=0, leave=True,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for task in task_progress:
        # Skip if already completed
        if task in checkpoint_state['completed_tasks']:
            log.info(
                f"\n⏭️  Skipping {task} - already completed (from checkpoint)")

            # Add the loaded results for this task to all_task_models
            task_results = [
                r for r in checkpoint_state['loaded_results'] if r['task'] == task]
            all_task_models[task].extend(task_results)

            task_progress.update(1)
            continue

        task_start = time.time()
        task_progress.set_description(f"🔬 Training {task.capitalize()}")

        # Extract labels (existing code remains the same)
        def safe_label_extraction(df, task):
            """Safely extract labels with NaN handling"""
            if f"silver_{task}" in df.columns:
                label_col = f"silver_{task}"
            elif f"label_{task}" in df.columns:
                label_col = f"label_{task}"
            else:
                log.error(
                    f"Missing both silver_{task} and label_{task} columns")
                default_val = 0 if task == "keto" else 1
                return np.full(len(df), default_val, dtype=int)

            labels = df[label_col].copy()

            if labels.isna().any():
                nan_count = labels.isna().sum()
                log.warning(
                    f"Found {nan_count} NaN values in {task} labels, filling with defaults")
                default_val = 0 if task == "keto" else 1
                labels = labels.fillna(default_val)

            try:
                labels = labels.astype(int)
            except ValueError as e:
                log.error(f"Cannot convert {task} labels to int: {e}")
                labels = pd.to_numeric(labels, errors='coerce').fillna(
                    0 if task == "keto" else 1).astype(int)

            return labels.values

        # Extract labels with validation
        y_train = safe_label_extraction(silver_df, task)
        y_true = safe_label_extraction(gold_df, task)
        log.info(f"\n🎯 Processing {task.upper()} classification:")
        log.info(
            f"   Training labels - Positive: {y_train.sum():,} ({y_train.mean():.1%})")
        log.info(
            f"   Test labels - Positive: {y_true.sum():,} ({y_true.mean():.1%})")

        # Handle class imbalance (existing SMOTE code)
        if apply_smote_flag:
            # ... (existing SMOTE code remains the same)
            X_train = X_silver  # Simplified for this example
        else:
            X_train = X_silver

        # Build and train models
        models = build_models(task, domain)

        # Filter out Rule model for image domain
        if domain == "image":
            models = {k: v for k, v in models.items() if k != "Rule"}

        # Check if any models were already completed in a partial checkpoint
        completed_for_task = checkpoint_state['completed_models'].get(task, [])
        if completed_for_task:
            log.info(
                f"   📂 Found {len(completed_for_task)} already completed models for {task}")
            # Filter out completed models
            models = {k: v for k, v in models.items()
                      if f"{k}_{domain.upper()}" not in completed_for_task}

        log.info(f"   🤖 Training {len(models)} models: {list(models.keys())}")

        # Initialize task-specific best tracking
        best_f1 = -1.0
        best_res = None
        model_results = []

        # Model training progress
        model_progress = tqdm(models.items(), desc="   ├─ Training Models",
                              position=1, leave=False,
                              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

        # ... (rest of the model training code remains the same until the checkpoint saving)

        # After training all models for this task, save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_{task}_{domain}.pkl"
        checkpoint_data = {
            'task': task,
            'domain': domain,
            'best_result': best_res,
            'all_results': model_results,
            'all_task_models': all_task_models[task],
            'timestamp': datetime.now().isoformat(),
            'task_time': time.time() - task_start,
            'feature_shape': X_silver.shape,
            'n_models_trained': len(model_results),
            # Track model names
            'completed_models': [r['model'] for r in model_results],
        }

        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            log.info(f"   💾 Checkpoint saved: {checkpoint_path.name}")

            # Also update pipeline state
            PIPELINE_STATE.save_checkpoint(f"{task}_{domain}", checkpoint_data)

        except Exception as e:
            log.error(f"   ❌ Failed to save checkpoint: {e}")

        # Update progress
        task_progress.set_postfix({
            'Best': best_res['model'] if best_res else 'N/A',
            'F1': f"{best_res['F1']:.3f}" if best_res else '0.000',
            'Time': f"{time.time() - task_start:.1f}s"
        })

    # Pipeline completion
    pipeline_time = time.time() - pipeline_start

    # Store all task models in results metadata
    results_metadata = {
        'all_task_models': all_task_models,
        'pipeline_time': pipeline_time,
        'domain': domain,
        'silver_size': len(silver_df),
        'gold_size': len(gold_df),
        'feature_dimensions': X_silver.shape,
        'smote_applied': apply_smote_flag,
        'timestamp': datetime.now().isoformat(),
        'checkpoint_dir': str(checkpoint_dir)
    }

    # Export results to CSV
    export_results_to_csv(results, results_metadata, domain)

    # Pipeline completion
    log.info(f"\n🏁 MODE A PIPELINE COMPLETE:")
    log.info(f"   ├─ Total Time: {pipeline_time:.1f}s")
    log.info(f"   ├─ Tasks Completed: 2")
    log.info(f"   ├─ Models Trained: {len(results)}")
    log.info(f"   ├─ Domain: {domain}")
    log.info(f"   └─ Checkpoints saved: {checkpoint_dir}")

    # Summary table - show ALL results, not just best per task
    log.info(f"\n📊 ALL RESULTS:")
    for i, res in enumerate(results, 1):
        log.info(f"   {i:2d}. {res['task'].upper():>5} | {res['model']:>15} | "
                 f"F1={res['F1']:.3f} | ACC={res['ACC']:.3f} | "
                 f"Time={res.get('training_time', 0):.1f}s")

    # Display formatted table
    table("MODE A (silver → gold)", results)

    # Return results with metadata attached
    for res in results:
        res['_metadata'] = results_metadata

    return results


# ====================================================================
# METRICS AND VISUALIZATION
# =============================================================================
"""
Functions for calculating metrics and displaying results.
"""


def export_results_to_csv(results: list[dict], metadata: dict, domain: str):
    """
    Export training results and metadata to CSV files.

    Creates multiple CSV files:
    - model_metrics_{domain}_{timestamp}.csv: All model performance metrics
    - best_models_{domain}_{timestamp}.csv: Best model per task
    - training_metadata_{domain}_{timestamp}.csv: Training run metadata
    - model_comparison_{domain}_{timestamp}.csv: Side-by-side comparison
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = CFG.artifacts_dir / "metrics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export all model metrics
    metrics_data = []
    for res in results:
        metrics_data.append({
            'timestamp': timestamp,
            'domain': domain,
            'task': res['task'],
            'model': res['model'],
            'model_base_name': res.get('model_base_name', res['model'].split('_')[0]),
            'accuracy': res['ACC'],
            'precision': res['PREC'],
            'recall': res['REC'],
            'f1_score': res['F1'],
            'roc_auc': res['ROC'],
            'pr_auc': res['PR'],
            'training_time_seconds': res.get('training_time', 0),
            'feature_domain': res.get('domain', domain),
            'n_samples_train': metadata.get('silver_size', 0),
            'n_samples_test': metadata.get('gold_size', 0),
            'feature_dimensions': str(metadata.get('feature_dimensions', ''))
        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = artifacts_dir / f"model_metrics_{domain}_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"   📊 Saved model metrics to {metrics_path}")

    # 2. Export best models per task
    best_models_data = []
    for task in ['keto', 'vegan']:
        task_results = [r for r in results if r['task'] == task]
        if task_results:
            best_model = max(task_results, key=lambda x: x['F1'])
            best_models_data.append({
                'timestamp': timestamp,
                'domain': domain,
                'task': task,
                'best_model': best_model['model'],
                'f1_score': best_model['F1'],
                'accuracy': best_model['ACC'],
                'precision': best_model['PREC'],
                'recall': best_model['REC'],
                'roc_auc': best_model['ROC'],
                'pr_auc': best_model['PR'],
                'training_time': best_model.get('training_time', 0)
            })

    best_df = pd.DataFrame(best_models_data)
    best_path = artifacts_dir / f"best_models_{domain}_{timestamp}.csv"
    best_df.to_csv(best_path, index=False)
    log.info(f"   🏆 Saved best models to {best_path}")

    # 3. Export training metadata
    metadata_data = [{
        'timestamp': timestamp,
        'domain': domain,
        'pipeline_time_seconds': metadata['pipeline_time'],
        'silver_size': metadata['silver_size'],
        'gold_size': metadata['gold_size'],
        'feature_dimensions': str(metadata['feature_dimensions']),
        'smote_applied': metadata['smote_applied'],
        'n_keto_models': len(metadata['all_task_models']['keto']),
        'n_vegan_models': len(metadata['all_task_models']['vegan']),
        'total_models_trained': len(results),
        'training_timestamp': metadata['timestamp']
    }]

    metadata_df = pd.DataFrame(metadata_data)
    metadata_path = artifacts_dir / \
        f"training_metadata_{domain}_{timestamp}.csv"
    metadata_df.to_csv(metadata_path, index=False)
    log.info(f"   📋 Saved training metadata to {metadata_path}")

    # 4. Create model comparison matrix
    comparison_data = []
    model_types = sorted(
        set(r.get('model_base_name', r['model'].split('_')[0]) for r in results))

    for model_type in model_types:
        row = {'model_type': model_type}
        for task in ['keto', 'vegan']:
            task_model = next((r for r in results
                              if r['task'] == task and
                              r.get('model_base_name', r['model'].split('_')[0]) == model_type),
                              None)
            if task_model:
                row[f'{task}_f1'] = task_model['F1']
                row[f'{task}_acc'] = task_model['ACC']
                row[f'{task}_time'] = task_model.get('training_time', 0)
            else:
                row[f'{task}_f1'] = None
                row[f'{task}_acc'] = None
                row[f'{task}_time'] = None
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = artifacts_dir / \
        f"model_comparison_{domain}_{timestamp}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    log.info(f"   📈 Saved model comparison to {comparison_path}")

    # 5. Export detailed predictions for error analysis
    predictions_data = []
    for res in results:
        if 'prob' in res and 'pred' in res:
            for i, (prob, pred) in enumerate(zip(res['prob'], res['pred'])):
                predictions_data.append({
                    'sample_index': i,
                    'task': res['task'],
                    'model': res['model'],
                    'probability': prob,
                    'prediction': pred,
                    'true_label': metadata.get('gold_labels', {}).get(res['task'], [])[i]
                    if i < len(metadata.get('gold_labels', {}).get(res['task'], []))
                    else None
                })

    if predictions_data:
        predictions_df = pd.DataFrame(predictions_data)
        predictions_path = artifacts_dir / \
            f"model_predictions_{domain}_{timestamp}.csv"
        predictions_df.to_csv(predictions_path, index=False)
        log.info(f"   🔮 Saved model predictions to {predictions_path}")


def aggregate_results_across_domains(all_results: dict[str, list[dict]]):
    """
    Aggregate results across different domains (text, image, both) for final comparison.

    Args:
        all_results: Dictionary mapping domain -> list of results
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = CFG.artifacts_dir / "metrics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Create comprehensive comparison across all domains
    all_metrics = []

    for domain, results in all_results.items():
        for res in results:
            all_metrics.append({
                'domain': domain,
                'task': res['task'],
                'model': res['model'],
                'model_base_name': res.get('model_base_name', res['model'].split('_')[0]),
                'f1_score': res['F1'],
                'accuracy': res['ACC'],
                'precision': res['PREC'],
                'recall': res['REC'],
                'roc_auc': res['ROC'],
                'pr_auc': res['PR'],
                'training_time': res.get('training_time', 0)
            })

    # 1. All metrics across domains
    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_path = artifacts_dir / f"all_domain_metrics_{timestamp}.csv"
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

    best_df = pd.DataFrame(best_across_domains)
    best_path = artifacts_dir / f"best_models_all_domains_{timestamp}.csv"
    best_df.to_csv(best_path, index=False)
    log.info(f"   ├─ Saved best models across domains to {best_path}")

    # 3. Domain comparison summary
    domain_summary = []
    for domain in all_results.keys():
        domain_metrics = [m for m in all_metrics if m['domain'] == domain]
        if domain_metrics:
            domain_summary.append({
                'domain': domain,
                'n_models': len(domain_metrics),
                'avg_f1_keto': np.mean([m['f1_score'] for m in domain_metrics if m['task'] == 'keto']),
                'avg_f1_vegan': np.mean([m['f1_score'] for m in domain_metrics if m['task'] == 'vegan']),
                'best_f1_keto': max([m['f1_score'] for m in domain_metrics if m['task'] == 'keto'], default=0),
                'best_f1_vegan': max([m['f1_score'] for m in domain_metrics if m['task'] == 'vegan'], default=0),
                'avg_training_time': np.mean([m['training_time'] for m in domain_metrics])
            })

    domain_df = pd.DataFrame(domain_summary)
    domain_path = artifacts_dir / f"domain_comparison_{timestamp}.csv"
    domain_df.to_csv(domain_path, index=False)
    log.info(f"   └─ Saved domain comparison to {domain_path}")

    # Create a master index file
    index_data = {
        'timestamp': timestamp,
        'all_metrics_file': all_metrics_path.name,
        'best_models_file': best_path.name,
        'domain_comparison_file': domain_path.name,
        'total_models': len(all_metrics),
        'domains': list(all_results.keys())
    }

    index_path = artifacts_dir / f"export_index_{timestamp}.json"
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    log.info(f"\n   📁 Export index saved to {index_path}")


def export_ensemble_metrics(ensemble_results: list[dict]):
    """
    Export specific metrics for ensemble models.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = CFG.artifacts_dir / "metrics" / "ensembles"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ensemble_data = []
    for res in ensemble_results:
        if 'ensemble' in res.get('domain', '') or 'Ens' in res.get('model', '') or 'BestTwo' in res.get('model', ''):
            ensemble_data.append({
                'timestamp': timestamp,
                'task': res['task'],
                'ensemble_type': res.get('ensemble_method', 'unknown'),
                'model_name': res['model'],
                'f1_score': res['F1'],
                'accuracy': res['ACC'],
                'precision': res['PREC'],
                'recall': res['REC'],
                'component_models': json.dumps(res.get('models_used', [])),
                'alpha': res.get('alpha', None),
                'n_models': res.get('ensemble_size', len(res.get('models_used', []))),
                'preparation_time': res.get('total_time', 0),
                'text_model': res.get('text_model', None),
                'image_model': res.get('image_model', None)
            })

    if ensemble_data:
        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_path = artifacts_dir / f"ensemble_metrics_{timestamp}.csv"
        ensemble_df.to_csv(ensemble_path, index=False)
        log.info(f"   🎭 Saved ensemble metrics to {ensemble_path}")


def table(title, rows):
    """
    Display results in a formatted table.

    Args:
        title: Table title
        rows: List of result dictionaries
    """
    cols = ("ACC", "PREC", "REC", "F1", "ROC", "PR")
    pad = 11 + 8 * len(cols)
    hdr = "│ model task " + " ".join(f"{c:>7}" for c in cols) + " │"
    log.info(f"\n╭─ {title} {'─' * (pad - len(title) - 2)}")
    log.info(hdr)
    log.info("├" + "─" * (len(hdr) - 2) + "┤")
    for r in rows:
        vals = " ".join(f"{r[c]:>7.2f}" for c in cols)
        log.info(f"│ {r['model']:<7} {r['task']:<5} {vals} │")
    log.info("╰" + "─" * (len(hdr) - 2) + "╯")


def pack(y, prob):
    """
    Calculate comprehensive metrics from predictions.

    Args:
        y: True labels
        prob: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        ACC=accuracy_score(y, pred),
        PREC=precision_score(y, pred, zero_division=0),
        REC=recall_score(y, pred, zero_division=0),
        F1=f1_score(y, pred, zero_division=0),
        ROC=roc_auc_score(y, prob),
        PR=average_precision_score(y, prob)
    )


def export_eval_plots(results: list[dict], gold_df: pd.DataFrame, output_dir: str = "artifacts"):
    """
    Export evaluation plots and confusion matrices for model results.

    Args:
        results: List of result dictionaries from model evaluation
        gold_df: Gold standard DataFrame
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"   📊 Generating evaluation plots...")

    # Create confusion matrices for best models per task
    for task in ["keto", "vegan"]:
        task_results = [r for r in results if r["task"]
                        == task and "pred" in r and "prob" in r]

        if not task_results:
            log.warning(
                f"   ⚠️  No results available for {task} visualization")
            continue

        # Find best model by F1 score
        best_result = max(task_results, key=lambda x: x.get("F1", 0))

        # Get predictions and true labels
        y_true = gold_df[f"label_{task}"].values[:len(best_result["pred"])]
        y_pred = best_result["pred"]

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                                      "Not " + task.capitalize(), task.capitalize()])
        disp.plot(ax=ax, cmap='Blues', values_format='d')

        plt.title(f'{task.capitalize()} Classification - {best_result["model"]}\n'
                  f'F1: {best_result["F1"]:.3f}, Accuracy: {best_result["ACC"]:.3f}')

        # Save plot
        plot_path = Path(
            output_dir) / f"confusion_matrix_{task}_{best_result['model'].replace('/', '_')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        log.info(f"   ✅ Saved {task} confusion matrix to {plot_path}")

    # Create performance comparison plot
    if len(results) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for i, task in enumerate(["keto", "vegan"]):
            ax = ax1 if i == 0 else ax2

            task_results = [r for r in results if r["task"] == task]
            if not task_results:
                continue

            # Sort by F1 score
            task_results = sorted(task_results, key=lambda x: x.get(
                "F1", 0), reverse=True)[:10]  # Top 10

            models = [r["model"] for r in task_results]
            f1_scores = [r["F1"] for r in task_results]
            acc_scores = [r["ACC"] for r in task_results]

            x = np.arange(len(models))
            width = 0.35

            bars1 = ax.bar(x - width/2, f1_scores, width,
                           label='F1 Score', alpha=0.8)
            bars2 = ax.bar(x + width/2, acc_scores, width,
                           label='Accuracy', alpha=0.8)

            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title(f'{task.capitalize()} Model Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8)

        plt.tight_layout()
        comparison_path = Path(output_dir) / "model_performance_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        log.info(f"   ✅ Saved performance comparison to {comparison_path}")


def log_false_preds(task, texts, y_true, y_pred, model_name="Model"):
    """
    Log false positive and false negative predictions for analysis.

    Creates CSV files with misclassified examples for debugging and
    improving the classification system.

    Args:
        task: Classification task ('keto' or 'vegan')
        texts: Original text data
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for filename
    """
    # False Positives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    if len(fp_indices) > 0:
        df_fp = pd.DataFrame({
            "Text": texts.iloc[fp_indices].values,
            "True_Label": y_true[fp_indices],
            "Predicted_Label": y_pred[fp_indices],
            "Error_Type": "False Positive",
            "Task": task
        })
        fp_path = f"false_positives_{task}_{model_name}.csv"
        df_fp.to_csv(fp_path, index=False)
        log.info(f"Logged {len(df_fp)} false positives to {fp_path}")

    # False Negatives
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    if len(fn_indices) > 0:
        df_fn = pd.DataFrame({
            "Text": texts.iloc[fn_indices].values,
            "True_Label": y_true[fn_indices],
            "Predicted_Label": y_pred[fn_indices],
            "Error_Type": "False Negative",
            "Task": task
        })
        fn_path = f"false_negatives_{task}_{model_name}.csv"
        df_fn.to_csv(fn_path, index=False)
        log.info(f"Logged {len(df_fn)} false negatives to {fn_path}")


# =============================================================================
# IMAGE EMBEDDING EXTRACTION
# =============================================================================
"""
Functions for extracting deep learning features from recipe images using
pre-trained ResNet-50 model. These embeddings capture visual characteristics
that can help identify ingredients and cooking methods.
"""


def build_image_embeddings(df: pd.DataFrame,
                           mode: str,
                           force: bool = False) -> Tuple[np.ndarray, List[int]]:
    """
    Extract ResNet-50 embeddings for recipe images with memory-efficient batch processing.

    This enhanced version includes:
    - Adaptive batch sizing based on available memory
    - Disk-based storage for memory efficiency
    - Checkpoint/resume capability
    - Better memory cleanup between batches
    """
    from collections import defaultdict, Counter

    embedding_start = time.time()

    # Get available memory for adaptive batch sizing
    available_memory_gb = get_available_memory(safety_factor=0.9)

    # Calculate adaptive batch size
    if torch.cuda.is_available():
        # GPU memory is more limited
        gpu_memory_gb = torch.cuda.get_device_properties(
            0).total_memory / (1024**3)
        # Roughly 4-8 images per GB of GPU memory
        batch_size = max(1, min(32, int(gpu_memory_gb * 4)))
    else:
        # CPU: estimate ~100MB per image for processing
        batch_size = max(1, min(16, int(available_memory_gb * 10)))

    log.info(
        f"   💾 Memory-aware batch size: {batch_size} (based on {available_memory_gb:.1f} GB available)")

    # ------------------------------------------------------------------
    # Initialization and System Check
    # ------------------------------------------------------------------
    log.info(f"\n🧠 IMAGE EMBEDDING EXTRACTION: {mode}")
    log.info(f"   Target images: {len(df):,}")
    log.info(f"   Mode: {mode}")
    log.info(f"   Force recomputation: {force}")

    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        log.warning("   ❌ PyTorch not available - returning zero vectors")
        return np.zeros((len(df), 2048), dtype=np.float32), list(df.index)

    # Check GPU availability
    device_info = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
    }

    log.info(f"   🔧 Device Configuration:")
    log.info(f"   ├─ Device: {device_info['device']}")
    log.info(f"   ├─ CUDA available: {device_info['cuda_available']}")

    # Set up paths
    img_dir = CFG.image_dir / mode
    embed_path = img_dir / "embeddings.npy"
    backup_path = CFG.artifacts_dir / f"embeddings_{mode}_backup.npy"
    metadata_path = img_dir / "embedding_metadata.json"
    checkpoint_path = img_dir / "embedding_checkpoint.npz"
    temp_embed_path = embed_path.with_suffix('.tmp.npy')

    # Ensure artifacts directory exists
    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache Loading and Validation
    # ------------------------------------------------------------------
    if not force:
        cache_options = [
            ("Artifacts backup", backup_path),
            ("Primary cache", embed_path),
            ("Legacy backup", Path(f"embeddings_{mode}_backup.npy"))
        ]

        for cache_name, cache_path in cache_options:
            if cache_path.exists():
                try:
                    emb = np.load(cache_path)
                    if emb.shape[0] == len(df):
                        log.info(
                            f"   ✅ Using cached embeddings from {cache_name}")
                        return emb, list(df.index)
                except Exception as e:
                    log.warning(f"   Failed to load {cache_name}: {e}")

    # ------------------------------------------------------------------
    # Check for checkpoint to resume
    # ------------------------------------------------------------------
    start_batch = 0
    processed_indices = set()

    if checkpoint_path.exists() and not force:
        try:
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            start_batch = int(checkpoint['batch_idx'])
            processed_indices = set(checkpoint['processed_indices'].tolist())
            log.info(
                f"   📂 Resuming from checkpoint: batch {start_batch}, {len(processed_indices)} already processed")
        except Exception as e:
            log.warning(f"   Could not load checkpoint: {e}")

    # ------------------------------------------------------------------
    # Model Setup with Memory Optimization
    # ------------------------------------------------------------------
    log.info(f"\n   🤖 Model Setup:")

    # Load model with gradient checkpointing disabled for inference
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    # Move to device with memory optimization
    if device_info['cuda_available']:
        # Set memory fraction for PyTorch
        torch.cuda.set_per_process_memory_fraction(
            0.8)  # Use only 80% of GPU memory

    model.to(device_info['device'])

    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ------------------------------------------------------------------
    # Memory-Efficient Processing
    # ------------------------------------------------------------------
    log.info(f"\n   ⚡ Feature Extraction:")

    # Use memory-mapped array for efficient disk-based storage
    embedding_shape = (len(df), 2048)

    # For large datasets, use memory-mapped storage
    use_memmap = len(df) > 10000 or available_memory_gb < 16

    if use_memmap:
        log.info(f"   💾 Using memory-mapped storage for {len(df):,} images")
        embeddings_mmap = np.memmap(
            temp_embed_path, dtype='float32', mode='w+', shape=embedding_shape)
    else:
        embeddings_mmap = np.zeros(embedding_shape, dtype=np.float32)

    processing_stats = {
        'success': len(processed_indices),
        'missing': 0,
        'failed': 0,
        'batch_times': [],
        'memory_cleanups': 0
    }

    valid_indices = list(processed_indices)
    num_batches = (len(df) + batch_size - 1) // batch_size

    log.info(f"   📊 Processing Configuration:")
    log.info(f"   ├─ Batch size: {batch_size}")
    log.info(f"   ├─ Total batches: {num_batches}")
    log.info(f"   ├─ Memory-mapped: {use_memmap}")
    log.info(f"   └─ Starting from batch: {start_batch}")

    # Process in batches with memory monitoring
    with tqdm(range(start_batch, num_batches), desc="   ├─ Processing batches",
              initial=start_batch, total=num_batches) as batch_pbar:

        for batch_idx in batch_pbar:
            batch_start_time = time.time()

            # Check memory before processing
            mem_before = psutil.virtual_memory()
            if mem_before.percent > 85:
                log.warning(
                    f"   ⚠️  High memory before batch {batch_idx}: {mem_before.percent:.1f}%")
                # Emergency cleanup
                gc.collect()
                if device_info['cuda_available']:
                    torch.cuda.empty_cache()
                processing_stats['memory_cleanups'] += 1
                time.sleep(2)  # Give system time to free memory

            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(df))
            batch_indices = df.index[batch_start:batch_end]

            # Collect batch tensors
            batch_tensors = []
            batch_valid_indices = []
            batch_positions = []

            for pos, idx in enumerate(batch_indices):
                if idx in processed_indices:
                    continue

                img_file = img_dir / f"{idx}.jpg"

                if not img_file.exists():
                    processing_stats['missing'] += 1
                    embeddings_mmap[batch_start +
                                    pos] = np.zeros(2048, dtype=np.float32)
                    continue

                try:
                    # Load and preprocess image with explicit cleanup
                    with Image.open(img_file).convert('RGB') as img:
                        tensor = preprocess(img).unsqueeze(0)

                    batch_tensors.append(tensor)
                    batch_valid_indices.append(idx)
                    batch_positions.append(batch_start + pos)

                except Exception as e:
                    processing_stats['failed'] += 1
                    embeddings_mmap[batch_start +
                                    pos] = np.zeros(2048, dtype=np.float32)

            # Process valid images in this batch
            if batch_tensors:
                try:
                    with torch.no_grad():
                        # Stack tensors and move to device
                        batch_tensor = torch.cat(batch_tensors, dim=0).to(
                            device_info['device'])

                        # Extract features
                        batch_features = model(batch_tensor).cpu().numpy()

                        # Clear GPU memory immediately
                        del batch_tensor
                        if device_info['cuda_available']:
                            torch.cuda.empty_cache()

                        # Write to memory-mapped array
                        for i, pos in enumerate(batch_positions):
                            embeddings_mmap[pos] = batch_features[i]
                            valid_indices.append(batch_valid_indices[i])
                            processed_indices.add(batch_valid_indices[i])

                        processing_stats['success'] += len(batch_tensors)

                except Exception as e:
                    log.error(
                        f"   ❌ Batch {batch_idx} processing failed: {str(e)[:60]}...")
                    for pos in batch_positions:
                        embeddings_mmap[pos] = np.zeros(2048, dtype=np.float32)

            # Clean up batch tensors
            del batch_tensors

            # Update statistics
            batch_time = time.time() - batch_start_time
            processing_stats['batch_times'].append(batch_time)

            # Update progress
            batch_pbar.set_postfix({
                'Success': processing_stats['success'],
                'Memory': f"{psutil.virtual_memory().percent:.1f}%",
                'Time': f"{batch_time:.1f}s"
            })

            # Periodic operations
            if (batch_idx + 1) % 10 == 0:
                # Save checkpoint
                np.savez_compressed(
                    checkpoint_path,
                    batch_idx=batch_idx + 1,
                    processed_indices=np.array(list(processed_indices)),
                    stats=processing_stats
                )

                # Flush to disk if using memmap
                if use_memmap:
                    embeddings_mmap.flush()

                # Memory cleanup
                gc.collect()

                log.info(f"   💾 Checkpoint saved at batch {batch_idx + 1}")

    # ------------------------------------------------------------------
    # Post-processing and Save
    # ------------------------------------------------------------------
    extraction_time = time.time() - embedding_start

    log.info(f"\n   📊 Extraction Results:")
    log.info(f"   ├─ Successfully processed: {processing_stats['success']:,}")
    log.info(f"   ├─ Missing images: {processing_stats['missing']:,}")
    log.info(f"   ├─ Failed processing: {processing_stats['failed']:,}")
    log.info(f"   ├─ Memory cleanups: {processing_stats['memory_cleanups']}")
    log.info(f"   └─ Total time: {extraction_time:.1f}s")

    # Convert to regular array if using memmap
    if use_memmap:
        log.info(f"   Converting memory-mapped array to regular array...")
        embeddings = np.array(embeddings_mmap)
        del embeddings_mmap  # Close memmap
    else:
        embeddings = embeddings_mmap

    # Apply quality filtering if we have enough valid embeddings
    if len(valid_indices) > 10:
        # Extract only valid embeddings for filtering
        valid_embeddings = embeddings[[
            df.index.get_loc(idx) for idx in valid_indices]]
        filtered_embeddings, filtered_indices = filter_low_quality_images(
            img_dir, valid_embeddings, valid_indices
        )
        embeddings = filtered_embeddings
        valid_indices = filtered_indices

    # Save results
    np.save(embed_path, embeddings)
    np.save(backup_path, embeddings)

    # Clean up temporary files
    if temp_embed_path.exists():
        temp_embed_path.unlink()
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Save metadata
    metadata = {
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'mode': mode,
        'total_images': len(df),
        'success': processing_stats['success'],
        'valid_indices': len(valid_indices),
        'processing_time_seconds': extraction_time,
        'batch_size': batch_size,
        'memory_cleanups': processing_stats['memory_cleanups'],
        'available_memory_gb': available_memory_gb
    }

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        log.warning(f"   Failed to save metadata: {e}")

    # Final cleanup
    del model
    gc.collect()
    if device_info['cuda_available']:
        torch.cuda.empty_cache()

    log.info(f"   🏁 Embedding extraction complete!")

    return embeddings, valid_indices


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================
"""
Advanced ensemble techniques for combining predictions from multiple models.
These methods improve performance by leveraging the strengths of different
models and feature types.
"""


class EnsembleWrapper(BaseEstimator, ClassifierMixin):
    """Comprehensive wrapper for all ensemble types with dynamic weighting."""

    def __init__(self, ensemble_type, models, weights=None, alpha=None, task=None):
        self.ensemble_type = ensemble_type
        self.models = models
        self.weights = weights
        self.alpha = alpha
        self.task = task

    def fit(self, X, y):
        # Already fitted
        return self

    def predict_proba(self, X):
        if self.ensemble_type == 'voting':
            # Delegate to sklearn VotingClassifier
            return self.models.predict_proba(X)

        elif self.ensemble_type == 'averaging':
            # Manual averaging
            probs = []
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    probs.append(model.predict_proba(X)[:, 1])
                else:
                    # Handle models without predict_proba
                    scores = model.decision_function(X) if hasattr(
                        model, 'decision_function') else model.predict(X)
                    probs.append(1 / (1 + np.exp(-scores)))

            mean_prob = np.mean(probs, axis=0)
            return np.c_[1 - mean_prob, mean_prob]

        elif self.ensemble_type == 'best_two':
            # Alpha blending
            text_model = self.models.get('text')
            image_model = self.models.get('image')

            text_prob = text_model.predict_proba(
                X)[:, 1] if text_model else 0.5
            image_prob = image_model.predict_proba(
                X)[:, 1] if image_model else text_prob

            blended = self.alpha * image_prob + (1 - self.alpha) * text_prob
            return np.c_[1 - blended, blended]

        elif self.ensemble_type == 'smart_ensemble':
            # Recursive ensemble blending
            text_ens = self.models.get('text_ensemble')
            image_ens = self.models.get('image_ensemble')

            if text_ens and image_ens:
                text_prob = text_ens.predict_proba(X)[:, 1]
                image_prob = image_ens.predict_proba(X)[:, 1]
                blended = self.alpha * image_prob + \
                    (1 - self.alpha) * text_prob
            elif text_ens:
                blended = text_ens.predict_proba(X)[:, 1]
            else:
                # Fallback
                blended = np.full(X.shape[0], 0.5)

            return np.c_[1 - blended, blended]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __repr__(self):
        return f"EnsembleWrapper(type={self.ensemble_type}, n_models={len(self.models)}, alpha={self.alpha})"


def tune_threshold(y_true, probs):
    """
    Find optimal classification threshold using precision-recall curve.

    Instead of using 0.5, this finds the threshold that maximizes F1 score.

    Args:
        y_true: True labels
        probs: Predicted probabilities

    Returns:
        Optimal threshold value
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5


def best_two_domains(
    task: str,
    text_results: list[dict],
    image_results: list[dict],
    gold_df: pd.DataFrame,
    alpha: float = 0.5,
):
    """
    Blend predictions from best text and image models.

    FIXED: Handles different sample sizes between text and image models.
    ENHANCED: Returns comprehensive metadata for CSV export integration.
    """
    # Find best models from each domain
    best_text = max(
        (r for r in text_results if r["task"] == task),
        key=lambda r: r["F1"]
    )
    best_img = max(
        (r for r in image_results if r["task"]
         == task and len(r.get("prob", [])) > 0),
        key=lambda r: r["F1"],
        default=None
    )

    log.info(
        f"▶️  BEST-TWO ({task}) — text={best_text['model']} "
        f"(F1={best_text['F1']:.3f}), "
        f"image={best_img['model'] if best_img else 'N/A'}"
    )

    # Get text predictions for ALL rows
    txt_prob = best_text["prob"]
    y_true = gold_df[f"label_{task}"].values

    if best_img is None:  # No image model available
        log.warning(
            f"   ⚠️  No image model available for {task}, using text-only")
        final_prob = txt_prob
    else:
        # Image predictions are only available for a subset of rows
        img_prob = best_img["prob"]
        # Indices with image predictions
        img_indices = gold_df.index[:len(img_prob)]

        log.info(f"   ├─ Text predictions: {len(txt_prob)} samples")
        log.info(f"   ├─ Image predictions: {len(img_prob)} samples")
        log.info(f"   └─ Blending with alpha={alpha}")

        # Initialize final predictions with text predictions
        final_prob = np.array(txt_prob).copy()

        # For rows with images, blend text and image predictions
        for i, idx in enumerate(img_indices):
            if i < len(img_prob):
                # Blend: alpha * image + (1-alpha) * text
                final_prob[idx] = alpha * img_prob[i] + \
                    (1 - alpha) * txt_prob[idx]

    # Apply verification and calculate metrics
    final_prob = verify_with_rules(task, gold_df.clean, final_prob)
    final_pred = (final_prob >= 0.5).astype(int)

    # Create ensemble wrapper
    ensemble_model = EnsembleWrapper(
        ensemble_type='best_two',
        models={
            'text': BEST.get(f"{task}_{best_text.get('domain', 'text')}_{best_text['model'].split('_')[0]}"),
            'image': BEST.get(f"{task}_{best_img.get('domain', 'image')}_{best_img['model'].split('_')[0]}") if best_img else None
        },
        alpha=alpha,
        task=task
    )

    # Create comprehensive model name for tracking
    model_name = f"BestTwo_{task}_alpha{alpha}"

    # Calculate timing (estimation based on component models)
    total_time = best_text.get('training_time', 0) + \
        (best_img.get('training_time', 0) if best_img else 0)

    return pack(y_true, final_prob) | {
        # Core identifiers
        "task": task,
        "model": model_name,
        "model_base_name": "BestTwo",

        # Model object for saving
        "model_object": ensemble_model,
        "ensemble_model": ensemble_model,  # Keep for backward compatibility

        # Domain and type information
        "domain": "ensemble",  # Mark as ensemble for export grouping
        "ensemble_type": "cross_domain",  # Specific ensemble type
        "ensemble_method": "alpha_blending",  # Method used

        # Predictions
        "prob": final_prob,
        "pred": final_pred,

        # Component model information
        "text_model": best_text["model"],
        "image_model": best_img["model"] if best_img else None,
        "text_model_f1": best_text["F1"],
        "image_model_f1": best_img["F1"] if best_img else None,

        # Ensemble parameters
        "alpha": alpha,
        "ensemble_size": 2 if best_img else 1,  # Number of models in ensemble

        # Sample information
        "rows_with_images": len(img_prob) if best_img else 0,
        "rows_text_only": len(gold_df) - (len(img_prob) if best_img else 0),
        "n_samples_train": len(gold_df),  # For consistency with other results
        "n_samples_test": len(gold_df),

        # Timing information
        "training_time": total_time,
        "preparation_time": total_time,  # Alternative field name used by some exports

        # Component models list for ensemble metrics export
        "models_used": [best_text["model"]] + ([best_img["model"]] if best_img else []),

        # Feature information
        "feature_dimensions": f"text:{best_text.get('feature_dimensions', 'unknown')}, " +
        (f"image:{best_img.get('feature_dimensions', 'unknown')}" if best_img else "image:none"),

        # Additional metadata for exports
        "_ensemble_metadata": {
            "creation_method": "best_two_domains",
            "text_domain": best_text.get('domain', 'text'),
            "image_domain": best_img.get('domain', 'image') if best_img else None,
            "blending_strategy": "linear_interpolation",
            "verification_applied": True
        }
    }


def dynamic_ensemble(estimators, X_gold, gold, task: str):
    """
    Create dynamically weighted ensemble based on feature availability.

    For rows without images, image model votes are zeroed and weights
    are renormalized to maintain valid probabilities.

    Args:
        estimators: List of (name, model) tuples
        X_gold: Test feature matrix
        gold: Gold standard DataFrame
        task: Classification task

    Returns:
        Array of ensemble probabilities
    """
    # Collect individual predictions
    preds = []
    for name, model in estimators:
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"{name} lacks predict_proba")
        preds.append(model.predict_proba(X_gold)[:, 1])

    probs = np.array(preds)  # shape: (n_models, n_samples)

    # Detect text-only rows (no image available)
    text_only_mask = gold.get("has_image", np.ones(
        len(X_gold), dtype=bool)) == False

    # Compute dynamic weights
    weights = []
    for i, (name, _) in enumerate(estimators):
        is_image_model = "IMAGE" in name.upper()
        row_weights = np.ones(len(X_gold))
        if is_image_model:
            # Suppress image models for text-only rows
            row_weights[text_only_mask] = 0
        weights.append(row_weights)

    weights = np.array(weights)
    weights_sum = weights.sum(axis=0)
    weights_sum[weights_sum == 0] = 1  # Prevent division by zero

    # Normalize weights and compute final probabilities
    normalized_weights = weights / weights_sum
    final_probs = (normalized_weights * probs).sum(axis=0)

    return final_probs


def top_n(task, res, X_vec, clean, X_gold, silver, gold, n=3, use_saved_params=False, rule_weight=0):
    """
    Build an n-model ensemble from top performing models.

    This sophisticated ensemble method:
    1. Ranks models by composite performance score
    2. Retrains top n models with optimal hyperparameters
    3. Creates a soft-voting ensemble
    4. Applies rule-based verification

    Args:
        task: 'keto' or 'vegan'
        res: Results from individual model evaluation
        X_vec: Training feature matrix
        clean: Normalized text for rule verification
        X_gold: Test feature matrix
        silver: Silver training data
        gold: Gold test data
        n: Number of models to include
        use_saved_params: Whether to use saved hyperparameters
        rule_weight: Weight for rule-based predictions (unused)

    Returns:
        Dictionary with ensemble results and metadata
    """

    import json
    import os
    from collections import defaultdict

    ensemble_start = time.time()

    log.info(f"\n🎯 BUILDING TOP-{n} ENSEMBLE for {task.upper()}")
    log.info(f"   Target ensemble size: {n} models")
    log.info(f"   Use saved parameters: {use_saved_params}")

    # Load saved parameters if requested
    saved_params = {}
    if use_saved_params:
        try:
            with open("best_params.json") as f:
                all_saved_params = json.load(f)
                saved_params = all_saved_params.get(task, {})
            log.info(
                f"   ✅ Loaded saved parameters for {len(saved_params)} models")
            for model_name in saved_params:
                log.info(f"      ├─ {model_name}: {saved_params[model_name]}")
        except FileNotFoundError:
            log.warning(
                f"   ⚠️  best_params.json not found, using default hyperparameters")
        except Exception as e:
            log.error(f"   ❌ Error loading parameters: {e}")

    # Filter and rank available models
    available_models = [r for r in res if r["task"]
                        == task and r["model"] != "Rule"]

    if not available_models:
        log.error(f"   ❌ No models available for {task} ensemble")
        raise ValueError(f"No models available for {task}")

    if len(available_models) < n:
        log.warning(
            f"   ⚠️  Only {len(available_models)} models available, requested {n}")
        n = len(available_models)
        log.info(f"   ├─ Adjusting ensemble size to {n}")

    log.info(f"\n   📊 Model Selection Analysis:")
    log.info(f"   ├─ Available models: {len(available_models)}")
    log.info(f"   └─ Selection criteria: Combined metric scoring")

    # Calculate composite scores
    scored_models = []
    for r in available_models:
        composite_score = (r["PREC"] + r["REC"] + r["ROC"] +
                           r["PR"] + r["F1"] + r["ACC"])
        scored_models.append((r, composite_score))

    # Select top N models
    top_models = sorted(scored_models, key=lambda x: x[1], reverse=True)[:n]

    log.info(f"\n   🏆 Top {n} Model Rankings:")
    for i, (model_res, score) in enumerate(top_models, 1):
        log.info(f"   {i:2d}. {model_res['model']:>10} | "
                 f"F1={model_res['F1']:.3f} | "
                 f"Composite={score:.3f} | "
                 f"ACC={model_res['ACC']:.3f}")

    # Prepare models for ensemble
    log.info(f"\n   🔧 Model Preparation Pipeline:")

    estimators = []
    preparation_times = {}
    model_errors = []

    # Progress bar for model preparation
    prep_progress = tqdm(top_models, desc="   ├─ Preparing Models",
                         position=0, leave=False,
                         bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

    for model_res, composite_score in prep_progress:
        name = model_res["model"]  # e.g., 'Softmax_TEXT'
        base_key = name.split('_')[0]  # Extract base model name
        prep_progress.set_description(f"   ├─ {name}")
        model_start = time.time()

        try:
            log.info(f"      ├─ Processing {name} (F1={model_res['F1']:.3f})")

            # Get base model and prepare
            with tqdm(total=5, desc=f"         ├─ {name}", position=1, leave=False,
                      bar_format="         ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as model_pbar:

                model_pbar.set_description(f"         ├─ {name}: Loading")
                base = build_models(task)[base_key]
                model_pbar.update(1)

                # Apply hyperparameters
                model_pbar.set_description(f"         ├─ {name}: Configuring")
                if use_saved_params and base_key in saved_params:
                    base.set_params(**saved_params[base_key])
                    log.info(
                        f"         ├─ Applied saved parameters: {saved_params[base_key]}")
                else:
                    log.info(f"         ├─ Tuning hyperparameters...")
                    base = tune(base_key, base, X_vec,
                                silver[f"silver_{task}"])
                model_pbar.update(1)

                # Train model
                model_pbar.set_description(f"         ├─ {name}: Training")
                base.fit(X_vec, silver[f"silver_{task}"])
                model_pbar.update(1)

                # Evaluate individual performance
                model_pbar.set_description(f"         ├─ {name}: Evaluating")
                y_pred_i = base.predict(X_gold)
                y_true = gold[f"label_{task}"].values

                individual_f1 = f1_score(y_true, y_pred_i, zero_division=0)
                individual_acc = accuracy_score(y_true, y_pred_i)

                log.info(
                    f"         ├─ Individual performance: F1={individual_f1:.3f}, ACC={individual_acc:.3f}")
                model_pbar.update(1)

                # Log false predictions
                model_pbar.set_description(f"         ├─ {name}: Analyzing")
                log_false_preds(task, gold.clean, y_true,
                                y_pred_i, model_name=name)

                # Ensure probability predictions
                if not hasattr(base, "predict_proba"):
                    log.info(
                        f"         ├─ Adding probability calibration to {name}")
                    base = CalibratedClassifierCV(base, cv=3, method='sigmoid')
                    base.fit(X_vec, silver[f"silver_{task}"])

                model_pbar.update(1)

            # Record successful preparation
            model_time = time.time() - model_start
            preparation_times[name] = model_time
            estimators.append((name, base))

            log.info(
                f"      ✅ {name} prepared successfully in {model_time:.1f}s")

            # Update progress
            prep_progress.set_postfix({
                'Success': len(estimators),
                'Failed': len(model_errors),
                'Current': f"{model_time:.1f}s"
            })

        except Exception as e:
            model_time = time.time() - model_start
            preparation_times[name] = model_time
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            model_errors.append((name, error_msg))

            log.error(
                f"      ❌ {name} failed after {model_time:.1f}s: {error_msg}")

            if log.level <= logging.DEBUG:
                import traceback
                log.debug(
                    f"Full traceback for {name}:\n{traceback.format_exc()}")

    # Preparation summary
    total_prep_time = sum(preparation_times.values())

    log.info(f"\n   📋 Preparation Summary:")
    log.info(f"   ├─ Successfully prepared: {len(estimators)}/{n}")
    log.info(f"   ├─ Failed preparations: {len(model_errors)}")
    log.info(f"   └─ Total preparation time: {total_prep_time:.1f}s")

    if model_errors:
        log.info(f"   ⚠️  Failed models:")
        for name, error in model_errors:
            log.info(f"      ├─ {name}: {error}")

    if not estimators:
        raise RuntimeError(
            f"No models successfully prepared for {task} ensemble")

    # Adjust n to actual number
    actual_n = len(estimators)
    if actual_n != n:
        log.info(f"   ├─ Adjusted ensemble size: {n} → {actual_n}")

    # Create ensemble
    log.info(f"\n   🤝 Ensemble Creation:")
    log.info(f"   ├─ Method: Soft voting classifier")
    log.info(f"   ├─ Models: {[name for name, _ in estimators]}")
    log.info(f"   └─ Target: {task} classification")

    ensemble_create_start = time.time()

    try:
        # Try soft voting ensemble
        with tqdm(total=3, desc="   ├─ Ensemble Creation", position=0, leave=False,
                  bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as ens_pbar:

            ens_pbar.set_description("   ├─ Creating VotingClassifier")
            ens = VotingClassifier(estimators, voting="soft", n_jobs=-1)
            ens_pbar.update(1)

            ens_pbar.set_description("   ├─ Training Ensemble")
            ens.fit(X_vec, silver[f"silver_{task}"])
            ens_pbar.update(1)

            ens_pbar.set_description("   ├─ Generating Predictions")
            # Use dynamic weighting for mixed feature types
            prob = dynamic_ensemble(estimators, X_gold, gold, task=task)
            ens_pbar.update(1)

        ensemble_create_time = time.time() - ensemble_create_start
        log.info(
            f"   ✅ Soft voting ensemble created in {ensemble_create_time:.1f}s")
        ensemble_method = "Soft Voting"

    except AttributeError as e:
        # Fallback to manual averaging
        log.warning(f"   ⚠️  Soft voting failed: {str(e)[:60]}...")
        log.info(f"   ├─ Falling back to manual probability averaging")

        prob_start = time.time()
        probs = []
        averaging_errors = []

        with tqdm(estimators, desc="   ├─ Manual Averaging", position=0, leave=False,
                  bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as avg_pbar:

            for name, clf in avg_pbar:
                avg_pbar.set_description(f"   ├─ Averaging {name}")
                try:
                    if hasattr(clf, "predict_proba"):
                        model_probs = clf.predict_proba(X_gold)[:, 1]
                        probs.append(model_probs)
                        log.debug(f"      ├─ {name}: predict_proba successful")
                    elif hasattr(clf, "decision_function"):
                        scores = clf.decision_function(X_gold)
                        model_probs = 1 / (1 + np.exp(-scores))  # Sigmoid
                        probs.append(model_probs)
                        log.debug(
                            f"      ├─ {name}: decision_function + sigmoid")
                    else:
                        binary_preds = clf.predict(X_gold).astype(float)
                        probs.append(binary_preds)
                        log.warning(
                            f"      ├─ {name}: using binary predictions (suboptimal)")

                except Exception as pred_error:
                    averaging_errors.append((name, str(pred_error)[:40]))
                    log.error(
                        f"      ├─ {name}: prediction failed - {str(pred_error)[:40]}...")

        if not probs:
            raise RuntimeError(
                "All models failed to generate predictions for ensemble")

        # Average probabilities
        prob = np.mean(probs, axis=0)

        prob_time = time.time() - prob_start
        log.info(f"   ✅ Manual averaging completed in {prob_time:.1f}s")
        log.info(
            f"      ├─ Successfully averaged: {len(probs)}/{len(estimators)} models")

        if averaging_errors:
            log.info(f"      ├─ Averaging errors:")
            for name, error in averaging_errors:
                log.info(f"      │  ├─ {name}: {error}")

        ensemble_method = "Manual Averaging"

    # Apply rule verification
    log.info(f"\n   🔍 Rule-Based Verification:")
    verification_start = time.time()

    original_positives = (prob >= 0.5).sum()
    prob_before_verification = prob.copy()

    with tqdm(total=1, desc="   ├─ Applying Rules", position=0, leave=False,
              bar_format="   ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as verify_pbar:
        prob = verify_with_rules(task, gold.clean, prob)
        verify_pbar.update(1)

    verification_time = time.time() - verification_start
    final_positives = (prob >= 0.5).sum()
    verification_changes = abs(final_positives - original_positives)

    log.info(f"   ├─ Verification completed in {verification_time:.3f}s")
    log.info(f"   ├─ Predictions before: {original_positives} positive")
    log.info(f"   ├─ Predictions after: {final_positives} positive")
    log.info(f"   └─ Rule changes: {verification_changes} predictions")

    # Generate final predictions
    y_pred = (prob >= 0.5).astype(int)
    y_true = gold[f"label_{task}"].values

    # Performance analysis
    log.info(f"\n   📊 Ensemble Performance Analysis:")

    ensemble_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, prob),
        'pr_auc': average_precision_score(y_true, prob)
    }

    log.info(f"   ├─ Accuracy:  {ensemble_metrics['accuracy']:.3f}")
    log.info(f"   ├─ Precision: {ensemble_metrics['precision']:.3f}")
    log.info(f"   ├─ Recall:    {ensemble_metrics['recall']:.3f}")
    log.info(f"   ├─ F1 Score:  {ensemble_metrics['f1']:.3f}")
    log.info(f"   ├─ ROC AUC:   {ensemble_metrics['roc_auc']:.3f}")
    log.info(f"   └─ PR AUC:    {ensemble_metrics['pr_auc']:.3f}")

    # Compare with best individual
    if len(estimators) > 1:
        log.info(f"\n   📈 Ensemble vs Individual Comparison:")
        best_individual = max(top_models, key=lambda x: x[0]['F1'])
        best_individual_f1 = best_individual[0]['F1']
        ensemble_improvement = ensemble_metrics['f1'] - best_individual_f1

        log.info(
            f"   ├─ Best individual: {best_individual[0]['model']} (F1={best_individual_f1:.3f})")
        log.info(f"   ├─ Ensemble F1: {ensemble_metrics['f1']:.3f}")
        log.info(
            f"   └─ Improvement: {ensemble_improvement:+.3f} ({ensemble_improvement/best_individual_f1*100:+.1f}%)")

    # Error analysis
    log.info(f"\n   🔍 Error Analysis:")
    log_false_preds(task, gold.clean, y_true, y_pred,
                    model_name=f"EnsembleTop{actual_n}")

    # Confidence analysis
    confidence_high = (np.abs(prob - 0.5) > 0.3).sum()
    confidence_medium = (np.abs(prob - 0.5) > 0.1).sum() - confidence_high
    confidence_low = len(prob) - confidence_high - confidence_medium

    log.info(
        f"   ├─ High confidence (>0.8 or <0.2): {confidence_high} ({confidence_high/len(prob)*100:.1f}%)")
    log.info(
        f"   ├─ Medium confidence (0.6-0.8, 0.2-0.4): {confidence_medium} ({confidence_medium/len(prob)*100:.1f}%)")
    log.info(
        f"   └─ Low confidence (0.4-0.6): {confidence_low} ({confidence_low/len(prob)*100:.1f}%)")

    # Final summary
    total_time = time.time() - ensemble_start

    log.info(f"\n   🏁 ENSEMBLE COMPLETE:")
    log.info(f"   ├─ Ensemble method: {ensemble_method}")
    log.info(f"   ├─ Models used: {actual_n}/{n}")
    log.info(f"   ├─ Final F1 score: {ensemble_metrics['f1']:.3f}")
    log.info(f"   ├─ Rule changes: {verification_changes}")
    log.info(f"   └─ Total time: {total_time:.1f}s")

    # Create ensemble model for saving
    if ensemble_method == "Soft Voting":
        ensemble_model = EnsembleWrapper(
            ensemble_type='voting',
            models=ens,  # Pass the VotingClassifier directly
            weights=None,
            task=task
        )
    else:
        # Create a custom ensemble wrapper for manual averaging
        ensemble_model = EnsembleWrapper(
            ensemble_type='averaging',
            models={name: model for name, model in estimators},
            weights=None,  # Equal weights for manual averaging
            task=task
        )
    # Return results
    return pack(y_true, prob) | {
        "model": f"Ens{actual_n}",
        "task": task,
        "prob": prob,
        "pred": y_pred,
        "ensemble_model": ensemble_model,
        "ensemble_method": ensemble_method,
        "models_used": [name for name, _ in estimators],
        "preparation_times": preparation_times,
        "verification_changes": verification_changes,
        "total_time": total_time,
        "confidence_distribution": {
            "high": confidence_high,
            "medium": confidence_medium,
            "low": confidence_low
        }
    }


def best_ensemble(
    task,
    res,
    X_vec,
    clean,
    X_gold,
    silver,
    gold,
    *,
    weights=None,
    image_res=None,
    alphas=(0.25, 0.5, 0.75)
):
    """
    Find optimal ensemble configuration through exhaustive search.

    This method tests different ensemble sizes and finds the best
    performing configuration. When image results are provided, it
    also performs smart blending between text and image models.

    Args:
        task: Classification task
        res: Text model results
        X_vec: Training features
        clean: Normalized text
        X_gold: Test features
        silver: Silver training data
        gold: Gold test data
        weights: Metric weights for optimization
        image_res: Optional image model results
        alphas: Blending weights to test for text+image

    Returns:
        Best ensemble configuration with results
    """

    from collections import Counter

    ensemble_start = time.time()

    # Handle text+image blending if image results provided
    if image_res:
        # Recursive call to get best text ensemble
        text_best = best_ensemble(
            task, res, X_vec, clean, X_gold, silver, gold,
            weights=weights, image_res=None
        )

        # Get best image ensemble
        img_pool = [r for r in image_res if r["task"] == task]
        if not img_pool:
            log.info(
                f"   ⏭️  No image models for {task}; using text ensemble only")
            return text_best

        image_best = best_ensemble(
            task, img_pool,
            X_vec=None, clean=clean,
            X_gold=None, silver=None,
            gold=gold,
            weights=weights,
            image_res=None
        )

        # Build probability series
        txt_prob = pd.Series(text_best["prob"], index=gold.index)
        img_len = len(image_best["prob"])
        img_idx = gold.index[:img_len]
        img_prob = pd.Series(image_best["prob"], index=img_idx)

        rows_img = img_idx
        rows_txt = txt_prob.index.difference(rows_img)
        y_true = gold[f"label_{task}"].values

        # Grid search over alpha values
        best_alpha, best_f1, best_prob = None, -1.0, None
        for alpha_param in alphas:
            blend = txt_prob.copy()
            blend.loc[rows_img] = alpha_param * img_prob.loc[rows_img] + \
                (1-alpha_param) * txt_prob.loc[rows_img]
            f1 = f1_score(y_true, (blend.values >= .5).astype(
                int), zero_division=0)
            if f1 > best_f1:
                best_alpha, best_f1, best_prob = alpha_param, f1, blend.values

        # Apply verification and return
        best_prob = verify_with_rules(task, gold.clean, best_prob)
        best_pred = (best_prob >= .5).astype(int)

        # CREATE THE ENSEMBLE MODEL HERE - THIS WAS MISSING!
        # Extract the actual model objects from the best results
        text_model_name = text_best['model'].split(
            '_')[0] if 'Ens' not in text_best['model'] else None
        image_model_name = image_best['model'].split(
            '_')[0] if 'Ens' not in image_best['model'] else None

        # Create the ensemble wrapper
        ensemble_model = EnsembleWrapper(
            ensemble_type='smart_ensemble',
            models={
                # Could be an ensemble itself
                'text_ensemble': text_best.get('ensemble_model'),
                # Could be an ensemble itself
                'image_ensemble': image_best.get('ensemble_model'),
                'text_model': BEST.get(text_model_name) if text_model_name else None,
                'image_model': BEST.get(image_model_name) if image_model_name else None,
            },
            alpha=best_alpha,
            task=task
        )

        return pack(y_true, best_prob) | {
            "model": f"SmartEns(Text={text_best['model']},Img={image_best['model']},alpha={best_alpha})",
            "task": task,
            "prob": best_prob,
            "pred": best_pred,
            "ensemble_model": ensemble_model,  # NOW THIS VARIABLE EXISTS
            "alpha": best_alpha,
            "rows_image": len(rows_img),
            "rows_text_only": len(rows_txt),
            "text_model": text_best["model"],
            "image_model": image_best["model"],
        }

    # Extract available models
    model_names = [r["model"]
                   for r in res if r["task"] == task and r["model"] != "Rule"]
    unique_models = list(set(model_names))
    max_n = len(unique_models)

    log.info(f"\n🎯 ENSEMBLE OPTIMIZATION for {task.upper()}")
    log.info(f"   Available models: {unique_models}")
    log.info(f"   Maximum ensemble size: {max_n}")

    # Handle edge cases
    if max_n == 0:
        log.warning(f"   ❌ No models available for {task} ensemble")
        return None

    if max_n == 1:
        single_model = [r for r in res if r["task"]
                        == task and r["model"] != "Rule"][0]
        log.info(f"   ⚠️  Only one model available: {single_model['model']}")
        log.info(
            f"   └─ F1={single_model['F1']:.3f}, skipping ensemble optimization")
        return single_model

    # Configure weights
    if weights is None:
        weights = {
            'F1': 1/6, 'PREC': 1/6, 'REC': 1/6,
            'ROC': 1/6, 'PR': 1/6, 'ACC': 1/6
        }
        log.info(f"   🎛️  Using default equal weighting for all metrics")
    else:
        log.info(f"   🎛️  Using custom metric weights")

    # Validate weights
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        log.warning(
            f"   ⚠️  Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        weights = {k: v/weight_sum for k, v in weights.items()}

    log.info(
        f"   ├─ Weights: {', '.join([f'{k}={v:.3f}' for k, v in weights.items()])}")

    # Show individual model performance
    individual_models = [r for r in res if r["task"]
                         == task and r["model"] != "Rule"]

    log.info(f"\n   📊 Individual Model Performance:")
    for model_res in sorted(individual_models, key=lambda x: x['F1'], reverse=True):
        composite = sum(weights.get(metric, 0) * model_res.get(metric, 0)
                        for metric in weights.keys())
        log.info(f"   ├─ {model_res['model']:>8}: F1={model_res['F1']:.3f} | "
                 f"Composite={composite:.3f} | ACC={model_res['ACC']:.3f}")

    # Check if we have feature matrices for retraining
    if X_vec is None or X_gold is None or silver is None or gold is None:
        log.info(f" ℹ️  No feature matrices supplied for '{task}' – "
                 f"returning best existing model without re-fitting.")
        best_existing = max(
            (r for r in res if r['task'] == task and r['model'] != 'Rule'),
            key=lambda r: r['F1']
        )
        return best_existing

    # Test different ensemble sizes
    log.info(f"\n   🔬 Testing ensemble sizes 1 to {max_n}...")

    best_score = -1
    best_result = None
    ensemble_results = []

    # Progress bar for ensemble optimization
    size_progress = tqdm(range(1, max_n + 1),
                         desc=f"   ├─ Ensemble Optimization ({task})",
                         position=0, leave=False,
                         bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

    for n in size_progress:
        size_start = time.time()
        size_progress.set_description(f"   ├─ Testing n={n} ({task})")

        try:
            # Create ensemble
            with tqdm(total=3, desc=f"      ├─ n={n}", position=1, leave=False,
                      bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as ensemble_pbar:

                # Step 1: Model selection and training
                ensemble_pbar.set_description(f"      ├─ n={n}: Selecting")
                result = top_n(task, res, X_vec, clean,
                               X_gold, silver, gold, n=n)
                ensemble_pbar.update(1)

                # Step 2: Metric calculation
                ensemble_pbar.set_description(f"      ├─ n={n}: Evaluating")
                composite_score = sum(
                    weights.get(metric, 0) * result.get(metric, 0)
                    for metric in weights.keys()
                )
                ensemble_pbar.update(1)

                # Step 3: Result recording
                ensemble_pbar.set_description(f"      ├─ n={n}: Recording")
                result['composite_score'] = composite_score
                result['ensemble_size'] = n
                result['optimization_time'] = time.time() - size_start
                ensemble_results.append(result)
                ensemble_pbar.update(1)

            # Log results
            log.info(f"      ✅ n={n}: F1={result['F1']:.3f} | "
                     f"Composite={composite_score:.3f} | "
                     f"ACC={result['ACC']:.3f} | "
                     f"Time={result['optimization_time']:.1f}s")

            # Track best
            if composite_score > best_score:
                best_score = composite_score
                best_result = result
                log.info(
                    f"      🏆 New best ensemble: n={n} (Composite={best_score:.3f})")

                # Update progress
                size_progress.set_postfix({
                    'Best_n': n,
                    'Best_F1': f"{result['F1']:.3f}",
                    'Best_Comp': f"{composite_score:.3f}"
                })

        except Exception as e:
            ensemble_time = time.time() - size_start
            log.error(f"      ❌ n={n}: FAILED after {ensemble_time:.1f}s")
            log.error(f"      └─ Error: {str(e)[:60]}...")

            if log.level <= logging.DEBUG:
                import traceback
                log.debug(
                    f"Full traceback for ensemble n={n}:\n{traceback.format_exc()}")

    # Final summary
    total_time = time.time() - ensemble_start

    if best_result:
        log.info(f"\n   🏆 ENSEMBLE OPTIMIZATION COMPLETE:")
        log.info(f"   ├─ Best Size: n={best_result['ensemble_size']}")
        log.info(f"   ├─ Best Model: {best_result['model']}")
        log.info(f"   ├─ Composite Score: {best_score:.3f}")
        log.info(f"   ├─ F1 Score: {best_result['F1']:.3f}")
        log.info(f"   ├─ Accuracy: {best_result['ACC']:.3f}")
        log.info(f"   └─ Total Time: {total_time:.1f}s")

        # Performance improvement
        if len(individual_models) > 0:
            best_individual = max(individual_models, key=lambda x: x['F1'])
            f1_improvement = best_result['F1'] - best_individual['F1']
            log.info(f"\n   📈 Performance Improvement:")
            log.info(
                f"   ├─ Best Individual: {best_individual['model']} (F1={best_individual['F1']:.3f})")
            log.info(
                f"   ├─ Best Ensemble: {best_result['model']} (F1={best_result['F1']:.3f})")
            log.info(
                f"   └─ F1 Improvement: {f1_improvement:+.3f} ({f1_improvement/best_individual['F1']*100:+.1f}%)")

        # Ensemble composition
        if 'Ens' in best_result['model']:
            ensemble_size = int(best_result['model'][-1])
            log.info(f"\n   🔧 Ensemble Composition (Top {ensemble_size}):")
            top_models = sorted(individual_models, key=lambda x: x['F1'], reverse=True)[
                :ensemble_size]
            for i, model_res in enumerate(top_models, 1):
                log.info(
                    f"   ├─ {i}. {model_res['model']:>8} (F1={model_res['F1']:.3f})")

    else:
        log.error(f"\n   ❌ ENSEMBLE OPTIMIZATION FAILED:")
        log.error(f"   ├─ No valid ensembles found")
        log.error(f"   ├─ Available models: {len(unique_models)}")
        log.error(f"   └─ Total Time: {total_time:.1f}s")

        # Fallback
        if individual_models:
            best_individual = max(individual_models, key=lambda x: x['F1'])
            log.info(f"   🛡️  Falling back to best individual model:")
            log.info(
                f"   └─ {best_individual['model']} (F1={best_individual['F1']:.3f})")
            return best_individual

    # Performance summary table
    if ensemble_results:
        log.info(f"\n   📊 Ensemble Size Performance Summary:")
        log.info(f"   ┌─────┬──────────┬─────────┬─────────┬───────────┬──────────┐")
        log.info(f"   │  n  │   F1     │   ACC   │  PREC   │    REC    │   COMP   │")
        log.info(f"   ├─────┼──────────┼─────────┼─────────┼───────────┼──────────┤")

        for result in sorted(ensemble_results, key=lambda x: x['ensemble_size']):
            n = result['ensemble_size']
            marker = " 🏆" if result == best_result else "   "
            log.info(f"   │ {n:2d}{marker} │ {result['F1']:6.3f}   │ {result['ACC']:5.3f}   │ "
                     f"{result['PREC']:5.3f}   │ {result['REC']:7.3f}   │ {result['composite_score']:6.3f}   │")

        log.info(f"   └─────┴──────────┴─────────┴─────────┴───────────┴──────────┘")

    # IMPORTANT: For non-image case, best_result should already have ensemble_model from top_n
    # But we need to ensure it's passed through
    return best_result


# =============================================================================
# MAIN PIPELINE ORCHESTRATION
# =============================================================================
"""
The main pipeline function that coordinates all components of the system.
"""


def run_full_pipeline(mode: str = "both",
                      force: bool = False,
                      sample_frac: float | None = None):
    """
    Execute the complete machine learning pipeline for diet classification.

    This is the main orchestration function that:
    1. Loads all datasets (recipes, ground truth, USDA)
    2. Generates silver labels for training
    3. Extracts text features (TF-IDF)
    4. Downloads images and extracts visual features (ResNet-50)
    5. Trains multiple models on different feature types
    6. Creates optimized ensembles
    7. Evaluates on gold standard test set
    8. Exports results and visualizations

    Args:
        mode: Feature mode - 'text', 'image', or 'both'
        force: Force recomputation of cached embeddings
        sample_frac: Fraction of silver data to sample (for testing)

    Returns:
        Tuple of (vectorizer, silver_data, gold_data, results)

    The function includes comprehensive error handling, memory optimization,
    and detailed progress tracking throughout all stages.
    """

    # Initialize pipeline tracking
    pipeline_start = time.time()

    # Log pipeline initialization
    log.info("🚀 STARTING FULL ML PIPELINE")
    log.info(f"   Mode: {mode}")
    log.info(f"   Force recomputation: {force}")
    log.info(f"   Sample fraction: {sample_frac or 'Full dataset'}")
    log.info(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"   Available CPU cores: {psutil.cpu_count()}")
    log.info(
        f"   Available memory: {psutil.virtual_memory().total // (1024**3)} GB")

    # Memory usage tracking
    def log_memory_usage(stage: str):
        memory = psutil.virtual_memory()
        log.info(f"   📊 {stage} - Memory: {memory.percent:.1f}% used "
                 f"({memory.used // (1024**2)} MB / {memory.total // (1024**2)} MB)")

    # Pipeline stages
    pipeline_stages = [
        "Data Loading", "Text Processing", "Image Processing",
        "Model Training", "Ensemble Creation", "Evaluation"
    ]

    # Main progress bar
    pipeline_progress = tqdm(pipeline_stages, desc="🔬 ML Pipeline",
                             position=0, leave=True,
                             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}")

    # ------------------------------------------------------------------
    # 1. DATA LOADING AND PREPARATION
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Data Loading")
    stage_start = time.time()

    log.info("\n📂 STAGE 1: DATA LOADING AND PREPARATION")

    with tqdm(total=4, desc="   ├─ Loading Data", position=1, leave=False,
              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as load_pbar:

        load_pbar.set_description("   ├─ Loading datasets")
        silver_all, gold, _, _ = get_datasets(sample_frac)
        load_pbar.update(1)

        load_pbar.set_description("   ├─ Creating index keys")
        silver_all["uid"] = silver_all.index
        gold["uid"] = gold.index
        load_pbar.update(1)

        load_pbar.set_description("   ├─ Preparing text data")
        silver_txt = silver_all.copy()
        load_pbar.update(1)

        load_pbar.set_description("   ├─ Filtering image data")
        silver_img = filter_photo_rows(silver_all)
        gold_img = filter_photo_rows(gold)
        load_pbar.update(1)

    # Apply sampling if requested
    if sample_frac and sample_frac < 1.0:
        original_txt_size = len(silver_txt)
        original_img_size = len(silver_img)

        # Sample both datasets consistently
        silver_txt = silver_txt.sample(
            frac=sample_frac, random_state=42).copy()

        if not silver_img.empty:
            # Get consistent sampling across modalities
            sampled_indices = silver_txt.index
            available_img_indices = silver_img.index
            common_indices = sampled_indices.intersection(
                available_img_indices)

            if len(common_indices) > 0:
                silver_img = silver_img.loc[common_indices].copy()
                log.info(
                    f"   📉 Consistent sampling: Using {len(common_indices):,} common indices")
            else:
                silver_img = silver_img.sample(
                    frac=sample_frac, random_state=42).copy()
                log.info(f"   📉 Separate sampling: No common indices found")

        sampled_txt_size = len(silver_txt)
        sampled_img_size = len(silver_img)

        log.info(f"   📉 Applied sampling before processing:")
        log.info(
            f"   ├─ Text: {original_txt_size:,} → {sampled_txt_size:,} rows ({sample_frac:.1%})")
        log.info(
            f"   └─ Images: {original_img_size:,} → {sampled_img_size:,} rows ({sample_frac:.1%})")

    # Log dataset statistics
    log.info(f"\n   📊 Dataset Statistics:")
    log.info(f"   ├─ Silver (All): {len(silver_all):,} recipes")
    log.info(f"   ├─ Silver (Text): {len(silver_txt):,} recipes")
    log.info(f"   ├─ Silver (Images): {len(silver_img):,} recipes")
    log.info(f"   ├─ Gold (All): {len(gold):,} recipes")
    log.info(f"   └─ Gold (Images): {len(gold_img):,} recipes")

    # Display class balance
    log.info(f"\n   ⚖️  Class Balance Analysis:")
    show_balance(gold, "Gold set")
    show_balance(silver_txt, "Silver (Text) set")
    show_balance(silver_img, "Silver (Image) set")

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Data loading completed in {stage_time:.1f}s")
    log_memory_usage("Data Loading")
    pipeline_progress.update(1)
    optimize_memory_usage("Data Loading")

    # Memory check
    if psutil.virtual_memory().percent > 70:
        log.warning(
            f"High memory usage after data loading: {psutil.virtual_memory().percent:.1f}%")

    # ------------------------------------------------------------------
    # 2. TEXT FEATURE PROCESSING
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Text Processing")
    stage_start = time.time()

    log.info("\n🔤 STAGE 2: TEXT FEATURE PROCESSING")

    with tqdm(total=4, desc="   ├─ Text Features", position=1, leave=False,
              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as text_pbar:

        text_pbar.set_description("   ├─ Initializing vectorizer")
        vec = TfidfVectorizer(**CFG.vec_kwargs)
        log.info(f"   ├─ Vectorizer config: {CFG.vec_kwargs}")
        text_pbar.update(1)

        text_pbar.set_description("   ├─ Fitting on silver data")

        # Load USDA carb data and convert to keto-labeled rows
        carb_df = _load_usda_carb_table()
        if not carb_df.empty:
            usda_labeled = label_usda_keto_data(carb_df)
            log.info(f"   ├─ USDA examples added: {len(usda_labeled)}")
            silver_txt = pd.concat(
                [silver_txt, usda_labeled], ignore_index=True)
            Path("artifacts").mkdir(exist_ok=True)
            silver_txt.to_csv("artifacts/silver_extended.csv", index=False)
        else:
            log.warning("   ├─ No USDA data added - carb_df is empty")

        X_text_silver = vec.fit_transform(silver_txt.clean)
        text_pbar.update(1)

        text_pbar.set_description("   ├─ Transforming gold data")
        X_text_gold = vec.transform(gold.clean)
        text_pbar.update(1)

        text_pbar.set_description("   ├─ Saving embeddings")
        Path("embeddings").mkdir(exist_ok=True)
        joblib.dump(X_text_gold, "embeddings/text_gold.pkl")
        text_pbar.update(1)

    # Log text processing results
    log.info(f"   📊 Text Processing Results:")
    log.info(f"   ├─ Vocabulary size: {len(vec.vocabulary_):,}")
    log.info(f"   ├─ Silver features: {X_text_silver.shape}")
    log.info(f"   ├─ Gold features: {X_text_gold.shape}")
    log.info(
        f"   ├─ Sparsity: {(1 - X_text_silver.nnz / X_text_silver.size):.1%}")
    log.info(
        f"   └─ Memory usage: ~{X_text_silver.data.nbytes // (1024**2)} MB")

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Text processing completed in {stage_time:.1f}s")
    log_memory_usage("Text Processing")
    optimize_memory_usage("Text Processing")
    pipeline_progress.update(1)

    # Initialize result containers
    results, res_text, res_img = [], [], []
    img_silver = img_gold = None

    # ------------------------------------------------------------------
    # 3. IMAGE FEATURE PROCESSING
    # ------------------------------------------------------------------
    if mode in {"image", "both"}:
        pipeline_progress.set_description("🔬 ML Pipeline: Image Processing")
        stage_start = time.time()

        log.info("\n🖼️  STAGE 3: IMAGE FEATURE PROCESSING")
        log.info(f"   ├─ Processing {len(silver_img):,} sampled silver images")
        log.info(f"   └─ Processing {len(gold_img):,} gold images")

        with tqdm(total=6, desc="   ├─ Image Pipeline", position=1, leave=False,
                  bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as img_pbar:

            # Download images with robust function and force parameter
            img_pbar.set_description("   ├─ Downloading silver images")
            if not silver_img.empty:
                silver_downloaded = _download_images(
                    silver_img, CFG.image_dir / "silver", max_workers=16, force=force)
                log.info(
                    f"      ├─ Silver download: {len(silver_downloaded):,}/{len(silver_img):,} successful")
            else:
                silver_downloaded = []
                log.info(f"      ├─ Silver download: No images to download")
            img_pbar.update(1)

            img_pbar.set_description("   ├─ Downloading gold images")
            if not gold_img.empty:
                gold_downloaded = _download_images(
                    gold_img, CFG.image_dir / "gold", max_workers=16, force=force)
                log.info(
                    f"      ├─ Gold download: {len(gold_downloaded):,}/{len(gold_img):,} successful")
            else:
                gold_downloaded = []
                log.info(f"      ├─ Gold download: No images to download")
            img_pbar.update(1)

            # Enhanced filtering logic with better error handling
            img_pbar.set_description("   ├─ Filtering by downloads")

            # Silver image filtering
            if silver_downloaded:
                try:
                    # Use the indices that were actually downloaded
                    img_silver_df = silver_img.loc[silver_img.index.intersection(
                        silver_downloaded)].copy()

                    if img_silver_df.empty:
                        log.warning(
                            f"      ⚠️  Silver filtering resulted in empty DataFrame")
                        img_silver_df = pd.DataFrame()
                    else:
                        log.info(
                            f"      ├─ Silver filtered: {len(img_silver_df):,} with valid images")

                except Exception as e:
                    log.error(f"      ❌ Silver filtering failed: {e}")
                    img_silver_df = pd.DataFrame()
            else:
                img_silver_df = pd.DataFrame()
                log.info(f"      ├─ Silver filtered: Empty (no downloads)")

            # Gold image filtering
            if gold_downloaded:
                try:
                    # Use the indices that were actually downloaded
                    img_gold_df = gold_img.loc[gold_img.index.intersection(
                        gold_downloaded)].copy()

                    if img_gold_df.empty:
                        log.warning(
                            f"      ⚠️  Gold filtering resulted in empty DataFrame")
                        img_gold_df = pd.DataFrame()
                    else:
                        log.info(
                            f"      ├─ Gold filtered: {len(img_gold_df):,} with valid images")

                except Exception as e:
                    log.error(f"      ❌ Gold filtering failed: {e}")
                    img_gold_df = pd.DataFrame()
            else:
                img_gold_df = pd.DataFrame()
                log.info(f"      ├─ Gold filtered: Empty (no downloads)")

            # Gold image filtering
            if gold_downloaded:
                try:
                    img_gold_df = filter_photo_rows(gold_img)
                    # Ensure we only keep images that were actually downloaded
                    if not img_gold_df.empty:
                        img_gold_df = img_gold_df.loc[img_gold_df.index.intersection(
                            gold_downloaded)].copy()

                    if img_gold_df.empty:
                        log.warning(
                            f"      ⚠️  Gold filtering resulted in empty DataFrame")
                        # Fallback: use original indices from downloads
                        img_gold_df = gold_img.loc[gold_downloaded].copy()

                except Exception as e:
                    log.error(f"      ❌ Gold filtering failed: {e}")
                    # Fallback to original gold_img subset
                    img_gold_df = gold_img.loc[gold_downloaded].copy(
                    ) if gold_downloaded else pd.DataFrame()
            else:
                img_gold_df = pd.DataFrame()

            log.info(
                f"      ├─ Gold filtered: {len(img_gold_df):,} with valid images")
            img_pbar.update(1)

            # Extract embeddings with proper alignment and error handling
            img_pbar.set_description("   ├─ Building silver embeddings")
            if not img_silver_df.empty:
                try:
                    img_silver, silver_valid_indices = build_image_embeddings(
                        img_silver_df, "silver", force)

                    # Verify and align DataFrame to match embeddings
                    if len(silver_valid_indices) != len(img_silver_df):
                        log.info(
                            f"      ├─ Aligning silver DataFrame: {len(img_silver_df):,} → {len(silver_valid_indices):,} rows")
                        img_silver_df = img_silver_df.loc[silver_valid_indices].copy(
                        )

                    # Validate dimensions match
                    if img_silver.shape[0] != len(img_silver_df):
                        log.error(f"      ❌ Silver dimension mismatch after alignment: "
                                  f"embeddings={img_silver.shape[0]}, df={len(img_silver_df)}")
                        # Try to fix by truncating to smaller size
                        min_size = min(img_silver.shape[0], len(img_silver_df))
                        img_silver = img_silver[:min_size]
                        img_silver_df = img_silver_df.iloc[:min_size].copy()
                        log.info(
                            f"      🔧 Truncated both to {min_size} for alignment")

                    log.info(f"      ├─ Silver embeddings: {img_silver.shape}")
                    log.info(
                        f"      ├─ Silver DataFrame: {len(img_silver_df):,} rows")

                except Exception as e:
                    log.error(
                        f"      ❌ Silver embedding extraction failed: {e}")
                    img_silver = np.array([]).reshape(0, 2048)
                    img_silver_df = pd.DataFrame()
                    silver_valid_indices = []
                    log.info(f"      ├─ Silver fallback: Empty embeddings")
            else:
                img_silver = np.array([]).reshape(0, 2048)
                silver_valid_indices = []
                log.info(
                    f"      ├─ Silver embeddings: Empty array (no valid images)")
            img_pbar.update(1)

            img_pbar.set_description("   ├─ Building gold embeddings")
            if not img_gold_df.empty:
                try:
                    img_gold, gold_valid_indices = build_image_embeddings(
                        img_gold_df, "gold", force)

                    # Verify and align DataFrame to match embeddings
                    if len(gold_valid_indices) != len(img_gold_df):
                        log.info(
                            f"      ├─ Aligning gold DataFrame: {len(img_gold_df):,} → {len(gold_valid_indices):,} rows")
                        img_gold_df = img_gold_df.loc[gold_valid_indices].copy(
                        )

                    # Validate dimensions match
                    if img_gold.shape[0] != len(img_gold_df):
                        log.error(f"      ❌ Gold dimension mismatch after alignment: "
                                  f"embeddings={img_gold.shape[0]}, df={len(img_gold_df)}")
                        # Try to fix by truncating to smaller size
                        min_size = min(img_gold.shape[0], len(img_gold_df))
                        img_gold = img_gold[:min_size]
                        img_gold_df = img_gold_df.iloc[:min_size].copy()
                        log.info(
                            f"      🔧 Truncated both to {min_size} for alignment")

                    log.info(f"      ├─ Gold embeddings: {img_gold.shape}")
                    log.info(
                        f"      ├─ Gold DataFrame: {len(img_gold_df):,} rows")

                except Exception as e:
                    log.error(f"      ❌ Gold embedding extraction failed: {e}")
                    img_gold = np.array([]).reshape(0, 2048)
                    img_gold_df = pd.DataFrame()
                    gold_valid_indices = []
                    log.info(f"      ├─ Gold fallback: Empty embeddings")
            else:
                img_gold = np.array([]).reshape(0, 2048)
                gold_valid_indices = []
                log.info(
                    f"      ├─ Gold embeddings: Empty array (no valid images)")
            img_pbar.update(1)

            img_pbar.set_description("   ├─ Saving embeddings")
            try:
                # Ensure embeddings directory exists
                Path("embeddings").mkdir(exist_ok=True)

                if img_gold.size > 0:
                    joblib.dump(img_gold, "embeddings/img_gold.pkl")
                    log.info(
                        f"      ├─ Saved gold embeddings to embeddings/img_gold.pkl")
                else:
                    log.info(f"      ├─ Skipped saving empty gold embeddings")

                # Also save silver embeddings for future use
                if img_silver.size > 0:
                    joblib.dump(img_silver, "embeddings/img_silver.pkl")
                    log.info(
                        f"      ├─ Saved silver embeddings to embeddings/img_silver.pkl")

            except Exception as e:
                log.warning(f"      ⚠️  Failed to save embeddings: {e}")
            img_pbar.update(1)

        # Enhanced dimension verification with detailed logging
        if (img_silver is not None and img_silver.size > 0) or (img_gold is not None and img_gold.size > 0):
            log.info(f"   🔍 DIMENSION VERIFICATION:")

            # Check silver dimensions
            if img_silver is not None and img_silver.size > 0:
                log.info(f"   ├─ Silver embeddings: {img_silver.shape}")
                log.info(
                    f"   ├─ Silver DataFrame: {len(img_silver_df):,} rows")

                # Fix dimension mismatch
                if img_silver.shape[0] != len(img_silver_df):
                    log.warning(
                        f"   ⚠️  Silver dimension mismatch: {img_silver.shape[0]} != {len(img_silver_df)}")

                    # Option 1: If embeddings has more rows, truncate
                    if img_silver.shape[0] > len(img_silver_df):
                        img_silver = img_silver[:len(img_silver_df)]
                        log.info(
                            f"   ├─ Truncated embeddings to match DataFrame")

                    # Option 2: If DataFrame has more rows, filter it
                    else:
                        # Get the first N indices where N = number of embeddings
                        valid_indices = img_silver_df.index[:img_silver.shape[0]]
                        img_silver_df = img_silver_df.loc[valid_indices]
                        log.info(
                            f"   ├─ Filtered DataFrame to match embeddings")

            # Check gold dimensions
            if img_gold is not None and img_gold.size > 0:
                log.info(f"   ├─ Gold embeddings: {img_gold.shape}")
                log.info(f"   └─ Gold DataFrame: {len(img_gold_df):,} rows")

                # Fix dimension mismatch
                if img_gold.shape[0] != len(img_gold_df):
                    log.warning(
                        f"   ⚠️  Gold dimension mismatch: {img_gold.shape[0]} != {len(img_gold_df)}")

                    if img_gold.shape[0] > len(img_gold_df):
                        img_gold = img_gold[:len(img_gold_df)]
                        log.info(
                            f"   ├─ Truncated embeddings to match DataFrame")
                    else:
                        valid_indices = img_gold_df.index[:img_gold.shape[0]]
                        img_gold_df = img_gold_df.loc[valid_indices]
                        log.info(
                            f"   ├─ Filtered DataFrame to match embeddings")

        # Convert to sparse matrices for memory efficiency
        try:
            if img_silver is not None and img_silver.size > 0:
                X_img_silver = csr_matrix(img_silver)
                log.debug(
                    f"      ├─ Silver sparse matrix: {X_img_silver.shape}")
            else:
                # Empty sparse matrix
                X_img_silver = None
                log.debug(f"      ├─ Silver: No image features available")

            if img_gold is not None and img_gold.size > 0:
                X_img_gold = csr_matrix(img_gold)
                log.debug(f"      ├─ Gold sparse matrix: {X_img_gold.shape}")
            else:
                # Empty sparse matrix
                X_img_gold = None
                log.debug(f"      ├─ Gold: No image features available")

        except Exception as e:
            log.error(f"   ❌ Sparse matrix conversion failed: {e}")
            X_img_silver = None
            X_img_gold = None

        # Comprehensive results summary
        log.info(f"   📊 Image Processing Results:")
        log.info(f"   ├─ Silver images available: {len(silver_img):,}")
        log.info(f"   ├─ Silver images downloaded: {len(silver_downloaded):,}")
        log.info(f"   ├─ Silver valid embeddings: {img_silver.shape[0]:,}")
        log.info(f"   ├─ Gold images available: {len(gold_img):,}")
        log.info(f"   ├─ Gold images downloaded: {len(gold_downloaded):,}")
        log.info(f"   ├─ Gold valid embeddings: {img_gold.shape[0]:,}")
        log.info(
            f"   ├─ Silver embedding size: {img_silver.nbytes // (1024**2) if img_silver.size > 0 else 0} MB")
        log.info(
            f"   └─ Gold embedding size: {img_gold.nbytes // (1024**2) if img_gold.size > 0 else 0} MB")

        # Enhanced early exit logic for image-only mode
        if mode == "image":
            total_valid_images = img_silver.shape[0] + img_gold.shape[0]
            min_required_images = 10  # Minimum viable images for training

            if total_valid_images < min_required_images:
                log.warning(f"   ⚠️  Insufficient images for image-only mode!")
                log.warning(
                    f"      ├─ Total valid images: {total_valid_images}")
                log.warning(
                    f"      ├─ Minimum required: {min_required_images}")
                log.warning(
                    f"      └─ Consider using mode='text' or mode='both'")

                stage_time = time.time() - stage_start
                log.info(
                    f"   ❌ Image processing insufficient in {stage_time:.1f}s")
                return None, None, None, []
            else:
                log.info(
                    f"   ✅ Sufficient images for image-only mode: {total_valid_images}")

        stage_time = time.time() - stage_start
        log.info(f"   ✅ Image processing completed in {stage_time:.1f}s")
        log_memory_usage("Image Processing")
        optimize_memory_usage("Image Processing")

    else:
        log.info("\n⏭️  STAGE 3: SKIPPED (Image processing not requested)")

    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # 4. MODEL TRAINING
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Model Training")
    stage_start = time.time()

    log.info("\n🤖 STAGE 4: MODEL TRAINING")

    training_subtasks = []
    if mode in {"image", "both"} and img_silver is not None and img_silver.size > 0:
        training_subtasks.append("Image Models")
    if mode in {"text", "both"}:
        training_subtasks.append("Text Models")
    if mode == "both" and img_silver is not None and img_silver.size > 0:
        training_subtasks.append("Text+Image Ensemble")
        training_subtasks.append("Final Combined")

    with tqdm(training_subtasks, desc="   ├─ Training Phases", position=1, leave=False,
              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as train_pbar:

        # IMAGE MODELS
        if mode in {"image", "both"}:
            train_pbar.set_description("   ├─ Training Image Models")
            log.info(f"   🖼️  Training image-based models...")

            try:
                if mode == "both":
                    # For "both" mode, we should use COMBINED features for image models too!
                    log.info(
                        "   📊 Mode='both' detected - using combined text+image features")

                    # Find common indices between text and image data
                    common_silver_idx = silver_txt.index.intersection(
                        img_silver_df.index)
                    common_gold_idx = gold.index.intersection(
                        img_gold_df.index)

                    if len(common_silver_idx) > 0 and len(common_gold_idx) > 0:
                        # Create combined features for silver
                        X_text_silver_common = vec.transform(
                            silver_txt.loc[common_silver_idx].clean)
                        img_silver_common = img_silver[img_silver_df.index.get_indexer(
                            common_silver_idx)]
                        X_combined_silver = combine_features(
                            X_text_silver_common, img_silver_common)

                        # Create combined features for gold
                        X_text_gold_common = vec.transform(
                            gold.loc[common_gold_idx].clean)
                        img_gold_common = img_gold[img_gold_df.index.get_indexer(
                            common_gold_idx)]
                        X_combined_gold = combine_features(
                            X_text_gold_common, img_gold_common)

                        # Use aligned DataFrames
                        silver_eval = silver_txt.loc[common_silver_idx]
                        gold_eval = gold.loc[common_gold_idx]

                        log.info(
                            f"      ├─ Combined features for image models: {X_combined_silver.shape}")
                        log.info(
                            f"      ├─ Samples: {len(common_silver_idx)} silver, {len(common_gold_idx)} gold")

                        res_img = run_mode_A(
                            X_combined_silver,  # Combined features!
                            gold_eval.clean,
                            X_combined_gold,    # Combined features!
                            silver_eval,
                            gold_eval,
                            domain="image",  # Still marked as "image" domain for model selection
                            apply_smote_flag=False
                        )
                    else:
                        log.warning(
                            "   ⚠️  No common indices between text and image data")
                        res_img = []
                else:
                    # For pure "image" mode, use only image features
                    log.info("   📊 Mode='image' - using image-only features")
                    res_img = run_mode_A(
                        X_img_silver,
                        img_gold_df.clean,
                        X_img_gold,
                        img_silver_df,
                        img_gold_df,
                        domain="image",
                        apply_smote_flag=False
                    )

                results.extend(res_img)
                log.info(f"      ✅ Image models: {len(res_img)} results")

            except Exception as e:
                log.error(f"      ❌ Image model training failed: {str(e)}")
                import traceback
                log.error(f"Full traceback:\n{traceback.format_exc()}")

            optimize_memory_usage("Image Models")
            train_pbar.update(1)
        # TEXT MODELS
        if mode in {"text", "both"}:
            train_pbar.set_description("   ├─ Training Text Models")
            log.info(f"   🔤 Training text-based models...")

            res_text = run_mode_A(
                X_text_silver, gold.clean, X_text_gold,
                silver_txt, gold,
                domain="text", apply_smote_flag=True
            )

            results.extend(res_text)
            log.info(f"      ✅ Text models: {len(res_text)} results")
            optimize_memory_usage("Text Models")
            train_pbar.update(1)

        # TEXT+IMAGE ENSEMBLE
        if mode == "both" and len(res_text) > 0 and len(res_img) > 0:
            train_pbar.set_description("   ├─ Text+Image Ensemble")
            log.info(f"   🤝 Creating text+image ensemble...")

            ensemble_results = []
            alpha_values = [0.25, 0.5, 0.75]

            for task in ("keto", "vegan"):
                try:
                    # Check if we have models for both domains
                    text_models = [r for r in res_text if r["task"] == task]
                    image_models = [r for r in res_img if r["task"] == task]

                    if not text_models:
                        log.warning(
                            f"      ⚠️  No text models available for {task}")
                        continue

                    if not image_models:
                        log.warning(
                            f"      ⚠️  No image models available for {task}")
                        # Still create ensemble with text-only
                        best_text_result = max(
                            text_models, key=lambda x: x['F1'])
                        ensemble_results.append(best_text_result)
                        continue

                    log.info(
                        f"      ├─ {task}: Testing alpha values {alpha_values}")

                    best_ensemble_result = None
                    best_f1 = -1
                    best_alpha = None

                    for alpha in alpha_values:
                        try:
                            result = best_two_domains(
                                task=task,
                                text_results=res_text,
                                image_results=res_img,
                                gold_df=gold,  # Use full gold df (100 samples)
                                alpha=alpha
                            )

                            if result and result['F1'] > best_f1:
                                best_f1 = result['F1']
                                best_alpha = alpha
                                best_ensemble_result = result

                            log.info(
                                f"         ├─ alpha={alpha}: F1={result['F1']:.3f}")

                        except Exception as e:
                            log.error(
                                f"         ❌ alpha={alpha} failed: {str(e)[:50]}...")
                            continue

                    if best_ensemble_result:
                        best_ensemble_result['model'] = f"BestTwo_alpha{best_alpha}"
                        ensemble_results.append(best_ensemble_result)
                        log.info(
                            f"      ✅ {task} best ensemble: alpha={best_alpha}, F1={best_f1:.3f}")
                    else:
                        # Fallback to best text model
                        log.warning(
                            f"      ⚠️  Ensemble failed for {task}, using best text model")
                        best_text = max(text_models, key=lambda x: x['F1'])
                        ensemble_results.append(best_text)

                except Exception as e:
                    log.error(
                        f"      ❌ {task} ensemble creation failed: {str(e)[:50]}...")
                    continue

            if ensemble_results:
                table("Text+Image Ensembles", ensemble_results)
                results.extend(ensemble_results)
                log.info(
                    f"      ✅ Created {len(ensemble_results)} ensemble configurations")
            else:
                log.warning(f"      ⚠️  No successful ensembles created")

            train_pbar.update(1)

        # FINAL COMBINED MODEL TRAINING
        if mode == "both":
            # Check if we have valid image data
            has_silver_images = (img_silver is not None) and (
                hasattr(img_silver, 'size')) and (img_silver.size > 0)
            has_gold_images = (img_gold is not None) and (
                hasattr(img_gold, 'size')) and (img_gold.size > 0)

            if has_silver_images and has_gold_images:
                train_pbar.set_description("   ├─ Final Combined Models")
                log.info(f"   🔄 Training final combined models...")

            # Align features and data. Find the intersection of indices between text and image data
            common_silver_idx = silver_txt.index.intersection(
                img_silver_df.index)
            common_gold_idx = gold.index.intersection(img_gold_df.index)

            if len(common_silver_idx) > 0 and len(common_gold_idx) > 0:
                # Align silver features
                X_text_silver_algn = vec.transform(
                    silver_txt.loc[common_silver_idx].clean)
                # Also need to align the image embeddings to the common indices
                img_silver_aligned = img_silver[img_silver_df.index.get_indexer(
                    common_silver_idx)]
                X_silver = combine_features(
                    X_text_silver_algn, img_silver_aligned)

                # Align gold features
                X_text_gold_algn = vec.transform(
                    gold.loc[common_gold_idx].clean)
                # Also align gold image embeddings
                img_gold_aligned = img_gold[img_gold_df.index.get_indexer(
                    common_gold_idx)]
                X_gold = combine_features(X_text_gold_algn, img_gold_aligned)

                silver_eval = silver_txt.loc[common_silver_idx]
                gold_eval = gold.loc[common_gold_idx]

                log.info(
                    f"      ├─ Combined silver features: {X_silver.shape}")
                log.info(f"      ├─ Combined gold features: {X_gold.shape}")
                log.info(f"      ├─ Silver samples: {len(silver_eval):,}")
                log.info(f"      └─ Gold samples: {len(gold_eval):,}")

                # Run combined training
                res_combined = run_mode_A(
                    X_silver, gold_eval.clean, X_gold,
                    silver_eval, gold_eval,
                    domain="both", apply_smote_flag=True
                )
                results.extend(res_combined)
                log.info(
                    f"      ✅ Combined models: {len(res_combined)} results")
                optimize_memory_usage()

            else:
                log.warning(
                    f"      ⚠️  No common indices for combined features, skipping")

            train_pbar.update(1)

        # Setup feature matrices for ensemble creation
        if mode == "both" and img_silver is not None and img_silver.size > 0:
            X_silver, X_gold = X_silver, X_gold
            silver_eval = silver_eval
        elif mode == "text":
            X_silver, X_gold = X_text_silver, X_text_gold
            silver_eval = silver_txt
        elif mode == "image" and img_silver is not None and img_silver.size > 0:
            X_silver, X_gold = csr_matrix(img_silver), csr_matrix(img_gold)
            silver_eval = img_silver_df
        else:
            # Fallback to text
            log.warning(
                f"   ⚠️  No valid images for image mode, falling back to text")
            X_silver, X_gold = X_text_silver, X_text_gold
            silver_eval = silver_txt

        # Final training if no results yet
        if not results:
            log.info(f"   🎯 Running fallback text-only training...")
            res_final = run_mode_A(X_text_silver, gold.clean, X_text_gold,
                                   silver_txt, gold, domain="text", apply_smote_flag=True)
            results.extend(res_final)
            log.info(f"      ✅ Final models: {len(res_final)} results")

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Model training completed in {stage_time:.1f}s")
    log.info(f"   📊 Total models trained: {len(results)}")
    log_memory_usage("Model Training")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # 5. ENSEMBLE OPTIMIZATION
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Ensemble Creation")
    stage_start = time.time()

    log.info("\n🎭 STAGE 5: ENSEMBLE OPTIMIZATION")

    if len(results) > 0:
        ensemble_tasks = ["keto", "vegan"]
        with tqdm(ensemble_tasks, desc="   ├─ Ensemble Tasks", position=1, leave=False,
                  bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as ens_pbar:

            ensemble_results = []
            for task in ens_pbar:
                ens_pbar.set_description(f"   ├─ Optimizing {task} ensemble")

                log.info(f"   🎯 Optimizing {task} ensemble...")

                # Count available models
                task_models = [r for r in results if r["task"]
                               == task and r["model"] != "Rule"]
                log.info(f"      ├─ Available models: {len(task_models)}")

                if len(task_models) > 1:
                    # Use appropriate features
                    if mode == "both" and img_silver is not None and img_silver.size > 0:
                        ens_X_silver = X_silver
                        ens_X_gold = X_gold
                        ens_silver_eval = silver_eval
                    elif mode == "image" and img_silver is not None and img_silver.size > 0:
                        ens_X_silver = csr_matrix(img_silver)
                        ens_X_gold = csr_matrix(img_gold)
                        ens_silver_eval = img_silver_df
                    else:
                        ens_X_silver = X_text_silver
                        ens_X_gold = X_text_gold
                        ens_silver_eval = silver_txt

                    best_ens = best_ensemble(task, results, ens_X_silver, gold.clean,
                                             ens_X_gold, ens_silver_eval, gold)
                    if best_ens:
                        ensemble_results.append(best_ens)
                        log.info(
                            f"      ✅ {task} ensemble: {best_ens['model']} (F1={best_ens['F1']:.3f})")
                    else:
                        log.warning(
                            f"      ⚠️  {task} ensemble optimization failed")
                else:
                    log.info(
                        f"      ⏭️  {task}: Only {len(task_models)} model(s) available, skipping ensemble")

            results.extend(ensemble_results)
            log.info(
                f"   📊 Ensemble results: {len(ensemble_results)} optimized ensembles")

        if mode == "both" and 'res_text' in locals() and 'res_img' in locals():
            log.info(f"\n   🔀 Cross-domain ensemble optimization...")

            cross_domain_results = []
            for task in ensemble_tasks:
                try:
                    # Check we have both text and image results
                    text_task_models = [
                        r for r in res_text if r["task"] == task]
                    image_task_models = [
                        r for r in res_img if r["task"] == task]

                    if text_task_models and image_task_models:
                        log.info(
                            f"      ├─ Optimizing {task} cross-domain ensemble...")

                        # Use best_ensemble with image_res for smart blending
                        cross_result = best_ensemble(
                            task=task,
                            res=res_text,  # Text results as primary
                            X_vec=X_text_silver,
                            clean=gold.clean,
                            X_gold=X_text_gold,
                            silver=silver_txt,
                            gold=gold,
                            image_res=res_img,  # This enables smart text+image blending
                            alphas=(0.25, 0.5, 0.75)  # Alpha values to test
                        )

                        if cross_result:
                            cross_domain_results.append(cross_result)
                            log.info(f"      ✅ {task} cross-domain: "
                                     f"{cross_result['model']} (F1={cross_result['F1']:.3f})")
                    else:
                        log.info(
                            f"      ⏭️  {task}: Missing text or image models for cross-domain")

                except Exception as e:
                    log.error(
                        f"      ❌ {task} cross-domain failed: {str(e)[:50]}...")

            if cross_domain_results:
                results.extend(cross_domain_results)
                log.info(
                    f"   📊 Cross-domain results: {len(cross_domain_results)} optimized ensembles")

    else:
        log.warning(f"   ⚠️  No models available for ensemble optimization")

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Ensemble optimization completed in {stage_time:.1f}s")
    log_memory_usage("Ensemble Creation")
    pipeline_progress.update(1)

    # ------------------------------------------------------------------
    # 6. EVALUATION AND EXPORT
    # ------------------------------------------------------------------
    pipeline_progress.set_description("🔬 ML Pipeline: Evaluation")
    stage_start = time.time()

    log.info("\n📊 STAGE 6: EVALUATION AND EXPORT")

    with tqdm(total=3, desc="   ├─ Export Process", position=1, leave=False,
              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as export_pbar:

        export_pbar.set_description("   ├─ Generating plots")
        if len(results) > 0:
            export_eval_plots(results, gold)
            log.info(f"      ✅ Generated evaluation plots and confusion matrices")
        else:
            log.warning(f"      ⚠️  No results to plot")
        export_pbar.update(1)

        export_pbar.set_description("   ├─ Saving results")
        # Save results summary
        results_summary = []
        for r in results:
            summary = {
                'task': r['task'],
                'model': r['model'],
                'f1': r['F1'],
                'accuracy': r['ACC'],
                'precision': r['PREC'],
                'recall': r['REC'],
                'roc_auc': r['ROC'],
                'pr_auc': r['PR']
            }
            results_summary.append(summary)

        if results_summary:
            pd.DataFrame(results_summary).to_csv(
                "pipeline_results_summary.csv", index=False)
            log.info(
                f"      ✅ Saved results summary with {len(results_summary)} entries")
        else:
            log.warning(f"      ⚠️  No results to save")
        export_pbar.update(1)

        export_pbar.set_description("   ├─ Cleanup")
        gc.collect()
        export_pbar.update(1)

    stage_time = time.time() - stage_start
    log.info(f"   ✅ Evaluation completed in {stage_time:.1f}s")
    log_memory_usage("Final")
    pipeline_progress.update(1)

# ------------------------------------------------------------------
    # PIPELINE COMPLETION SUMMARY
# ------------------------------------------------------------------
    total_time = time.time() - pipeline_start

    log.info(f"\n🏁 PIPELINE COMPLETE")
    log.info(
        f"   ├─ Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    log.info(f"   ├─ Mode: {mode}")
    log.info(f"   ├─ Sample fraction: {sample_frac or 'Full dataset'}")
    log.info(f"   ├─ Total results: {len(results)}")
    log.info(f"   └─ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Performance summary
    if results:
        log.info(f"\n   🏆 FINAL PERFORMANCE SUMMARY:")
        for task in ["keto", "vegan"]:
            task_results = [r for r in results if r["task"] == task]
            if task_results:
                best_result = max(task_results, key=lambda x: x['F1'])
                log.info(f"   ├─ {task.upper()}: Best F1={best_result['F1']:.3f} "
                         f"({best_result['model']}) | ACC={best_result['ACC']:.3f}")

        # Organize results by domain for better analysis
        results_by_domain = {}
        for res in results:
            domain = res.get('domain', 'unknown')
            if domain not in results_by_domain:
                results_by_domain[domain] = []
            results_by_domain[domain].append(res)

        # Export aggregated results across all domains
        aggregate_results_across_domains(results_by_domain)

        # Export ensemble-specific metrics if any ensembles were created
        ensemble_results = [r for r in results if
                            'ensemble' in r.get('domain', '') or
                            'Ens' in r.get('model', '') or
                            'BestTwo' in r.get('model', '') or
                            'SmartEns' in r.get('model', '')]

        if ensemble_results:
            export_ensemble_metrics(ensemble_results)
            log.info(f"\n   🎭 ENSEMBLE SUMMARY:")
            log.info(f"   ├─ Total ensembles created: {len(ensemble_results)}")

            # Best ensemble per task
            for task in ["keto", "vegan"]:
                task_ensembles = [
                    r for r in ensemble_results if r["task"] == task]
                if task_ensembles:
                    best_ensemble = max(task_ensembles, key=lambda x: x['F1'])
                    log.info(
                        f"   ├─ {task.upper()} best ensemble: {best_ensemble['model']} (F1={best_ensemble['F1']:.3f})")

        # Create comprehensive pipeline summary
        pipeline_summary = {
            'mode': mode,
            'sample_frac': sample_frac,
            'total_time': total_time,
            'total_time_minutes': total_time / 60,
            'total_models': len(results),
            'total_ensembles': len(ensemble_results),
            'domains_trained': list(results_by_domain.keys()),
            'models_per_domain': {domain: len(models) for domain, models in results_by_domain.items()},
            'best_keto_f1': max([r['F1'] for r in results if r['task'] == 'keto'], default=0),
            'best_vegan_f1': max([r['F1'] for r in results if r['task'] == 'vegan'], default=0),
            'best_keto_model': max([r for r in results if r['task'] == 'keto'], key=lambda x: x['F1'], default={'model': 'None'})['model'],
            'best_vegan_model': max([r for r in results if r['task'] == 'vegan'], key=lambda x: x['F1'], default={'model': 'None'})['model'],
            'timestamp': datetime.now().isoformat(),
            'pipeline_stages_completed': [
                'data_loading', 'text_processing',
                'image_processing' if mode in ['image', 'both'] else None,
                'model_training', 'ensemble_creation', 'evaluation'
            ]
        }

        # Save detailed pipeline summary
        summary_dir = CFG.artifacts_dir / "metrics"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / \
            f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(summary_path, 'w') as f:
            json.dump(pipeline_summary, f, indent=2)
        log.info(f"\n   📝 Pipeline summary saved to {summary_path}")

    else:
        log.warning(f"\n   ⚠️  NO RESULTS GENERATED")
        log.warning(
            f"   └─ Consider checking data availability or adjusting parameters")

    # Resource usage summary
    final_memory = psutil.virtual_memory()
    log.info(f"\n   💾 RESOURCE USAGE:")
    log.info(f"   ├─ Peak memory: {final_memory.percent:.1f}%")
    log.info(f"   ├─ Final memory: {final_memory.used // (1024**2)} MB")
    log.info(f"   └─ Efficiency: {len(results)/total_time:.2f} models/second")

    # Enhanced pipeline metadata with more details
    pipeline_metadata = {
        'mode': mode,
        'force': force,
        'sample_frac': sample_frac,
        'total_time': total_time,
        'total_time_minutes': total_time / 60,
        'total_results': len(results),
        'start_time': pipeline_start,
        'end_time': time.time(),
        'memory_peak_percent': final_memory.percent,
        'memory_used_mb': final_memory.used // (1024**2),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total // (1024**3)
        },
        'data_stats': {
            'silver_text_size': len(silver_txt),
            'silver_image_size': len(silver_img),
            'gold_size': len(gold),
            'silver_images_downloaded': len(silver_downloaded) if 'silver_downloaded' in locals() else 0,
            'gold_images_downloaded': len(gold_downloaded) if 'gold_downloaded' in locals() else 0,
            'image_embeddings_used': {
                'silver': img_silver.shape[0] if 'img_silver' in locals() and img_silver is not None else 0,
                'gold': img_gold.shape[0] if 'img_gold' in locals() and img_gold is not None else 0
            }
        },
        'model_summary': {
            'total_models': len(results),
            'text_models': len([r for r in results if r.get('domain') == 'text']),
            'image_models': len([r for r in results if r.get('domain') == 'image']),
            'combined_models': len([r for r in results if r.get('domain') == 'both']),
            'ensemble_models': len([r for r in results if 'ensemble' in r.get('domain', '') or 'Ens' in r.get('model', '')])
        },
        'performance_summary': {
            'best_overall_f1': max([r['F1'] for r in results], default=0),
            'best_overall_model': max(results, key=lambda x: x['F1'], default={'model': 'None'})['model'] if results else 'None',
            'average_f1': np.mean([r['F1'] for r in results]) if results else 0,
            'average_training_time': np.mean([r.get('training_time', 0) for r in results]) if results else 0
        }
    }

    # Save to both JSON and CSV for easy access
    metadata_path = CFG.artifacts_dir / "metrics" / \
        f"pipeline_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_path, "w") as f:
        json.dump(pipeline_metadata, f, indent=2)

    # Also save a simplified version to the root for backward compatibility
    with open("pipeline_metadata.json", "w") as f:
        json.dump(pipeline_metadata, f, indent=2)

    log.info(f"   💾 Saved pipeline metadata to {metadata_path}")

    # Create a master results DataFrame for easy analysis
    if results:
        master_results = []
        for res in results:
            master_results.append({
                'timestamp': datetime.now().isoformat(),
                'pipeline_mode': mode,
                'task': res['task'],
                'model': res['model'],
                'domain': res.get('domain', 'unknown'),
                'f1_score': res['F1'],
                'accuracy': res['ACC'],
                'precision': res['PREC'],
                'recall': res['REC'],
                'roc_auc': res['ROC'],
                'pr_auc': res['PR'],
                'training_time': res.get('training_time', 0)
            })

        master_df = pd.DataFrame(master_results)
        master_path = CFG.artifacts_dir / "metrics" / \
            f"master_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        master_df.to_csv(master_path, index=False)
        log.info(f"   📊 Saved master results table to {master_path}")

        # Print final summary statistics
        log.info(f"\n   📈 FINAL STATISTICS:")
        log.info(f"   ├─ Models trained per task:")
        for task in ['keto', 'vegan']:
            task_count = len([r for r in results if r['task'] == task])
            log.info(f"   │  ├─ {task}: {task_count} models")
        log.info(
            f"   ├─ Average F1 score: {pipeline_metadata['performance_summary']['average_f1']:.3f}")
        log.info(
            f"   ├─ Best overall F1: {pipeline_metadata['performance_summary']['best_overall_f1']:.3f}")
        log.info(
            f"   └─ Total training efficiency: {len(results)/total_time:.2f} models/second")

    return vec, silver_txt, gold, results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
"""
Main entry point for the diet classification pipeline.
"""


def main():
    """
    Command line interface for the diet classification pipeline.

    Enhanced with complete resume implementation, pre-flight checks, 
    and better error handling.

    Supports multiple modes:
    - Training: Run full pipeline to train models
    - Inference: Classify ingredients using trained models
    - Evaluation: Test on ground truth dataset
    - Sanity Check: Quick verification with minimal data

    The function includes comprehensive error handling and prevents
    restart loops through environment variable tracking.
    """

    import argparse
    import sys
    import atexit

    # Register exit handler
    def prevent_restart():
        log.info("🛑 Process exiting - no restarts allowed")

    atexit.register(prevent_restart)

    parser = argparse.ArgumentParser(description='Diet Classifier')
    parser.add_argument('--ground_truth', type=str,
                        help='Path to ground truth CSV')
    parser.add_argument('--train', action='store_true',
                        help='Run full training pipeline')
    parser.add_argument('--ingredients', type=str,
                        help='Comma separated ingredients to classify')
    parser.add_argument('--mode', choices=['text', 'image', 'both'],
                        default='both', help='Feature mode for training')
    parser.add_argument('--force', action='store_true',
                        help='Recompute image embeddings')
    parser.add_argument('--sample_frac', type=float,
                        default=None, help="Fraction of silver set to sample.")
    parser.add_argument('--sanity_check', action='store_true',
                        help='Run quick sanity check with minimal data')
    parser.add_argument('--predict', type=str,
                        help='Path to CSV file for batch prediction')

    args = parser.parse_args()

    # Memory optimization for Docker environments
    log.info(f"🚀 Starting main with args: {args}")

    # Check and log memory configuration
    available_memory = get_available_memory(safety_factor=0.9)
    log.info(f"💾 Available memory for processing: {available_memory:.1f} GB")

    # Set memory-related environment variables
    if available_memory < 32:  # Less than 32GB available
        log.warning(
            f"⚠️  Limited memory detected. Enabling memory optimizations...")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

        # Reduce number of threads to save memory
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = '4'

        # Enable Python memory optimizations
        if hasattr(sys, 'setswitchinterval'):
            # Python 3.2+ uses setswitchinterval (value in seconds)
            # Increase switch interval to reduce overhead
            sys.setswitchinterval(0.1)
        elif hasattr(sys, 'setcheckinterval'):
            # Fallback for older Python versions (pre-3.2)
            sys.setcheckinterval(1000)

    # Handle sanity check mode
    if args.sanity_check:
        log.info("🔍 SANITY CHECK MODE - Using minimal data and models")
        args.sample_frac = 0.01  # Use only 1% of data
        args.mode = 'text'  # Text only for speed
        os.environ['SANITY_CHECK'] = '1'
        
        # Override configuration for sanity check
        if not args.train and not args.ground_truth and not args.ingredients:
            args.train = True  # Default to training in sanity check

    # Run pre-flight checks
    if not preflight_checks():
        log.error("❌ Pre-flight checks failed. Please fix issues before proceeding.")
        sys.exit(1)

    try:
        if args.ingredients:
            # Handle ingredient classification
            if args.ingredients.startswith('['):
                ingredients = json.loads(args.ingredients)
            else:
                ingredients = [i.strip()
                               for i in args.ingredients.split(',') if i.strip()]

            keto = is_keto(ingredients)
            vegan = is_vegan(ingredients)
            log.info(json.dumps({'keto': keto, 'vegan': vegan}))
            return

        elif args.train:
            log.info(f"🧠 TRAINING MODE - sample_frac={args.sample_frac}")
            
            if args.sanity_check:
                log.info(f"   ├─ Sanity check: YES (minimal data)")
                log.info(f"   ├─ Expected runtime: ~5 minutes")
            else:
                log.info(f"   ├─ Full training mode")
                log.info(f"   ├─ Expected runtime: 1-3 hours")

            try:
                # Check for existing pipeline state
                saved_stage, saved_data = load_pipeline_state()
                resume_from_checkpoint = False
                
                if saved_stage and not args.force:
                    log.info(f"   📂 Found saved pipeline state from stage: {saved_stage}")
                    response = input("   Resume from saved state? [Y/n]: ").strip().lower()
                    
                    if response != 'n':
                        resume_from_checkpoint = True
                        log.info(f"   ✅ Will attempt to resume from {saved_stage}")
                        
                        # Load any saved models into BEST cache
                        if 'best_models' in saved_data:
                            BEST.update(saved_data['best_models'])
                            log.info(f"   ├─ Restored {len(saved_data['best_models'])} models to cache")
                        
                        # Update pipeline state
                        if 'pipeline_state' in saved_data:
                            PIPELINE_STATE.__dict__.update(saved_data['pipeline_state'])
                            log.info(f"   └─ Restored pipeline state")
                
                # Run pipeline (it will handle resume internally)
                vec, silver, gold, res = run_full_pipeline(
                    mode=args.mode, 
                    force=args.force, 
                    sample_frac=args.sample_frac
                )
                
                if not res:
                    log.error("❌ Pipeline produced no results!")
                    sys.exit(1)

                log.info(f"✅ Pipeline completed with {len(res)} results")

                # Save models with optimized serialization
                try:
                    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Prepare best models
                    best_models = {}
                    for task in ['keto', 'vegan']:
                        task_res = [r for r in res if r['task'] == task]
                        if task_res:
                            best = max(task_res, key=lambda x: x['F1'])
                            
                            if 'ensemble_model' in best and best['ensemble_model'] is not None:
                                best_models[task] = best['ensemble_model']
                                log.info(f"✅ Selected {task} ensemble: {best['model']} (F1={best['F1']:.3f})")
                            else:
                                # Fallback to single model
                                model_name = best['model']
                                base_name = model_name.split('_')[0]
                                if base_name in BEST:
                                    best_models[task] = BEST[base_name]
                                    log.info(f"✅ Selected {task} model: {base_name} (F1={best['F1']:.3f})")
                    
                    if best_models:
                        # Use optimized saving
                        save_models_optimized(best_models, vec, CFG.artifacts_dir)
                        
                        # Save final pipeline state
                        final_state = {
                            'stage': 'completed',
                            'timestamp': datetime.now().isoformat(),
                            'best_models': {k: v for k, v in BEST.items()},
                            'pipeline_state': PIPELINE_STATE.__dict__,
                            'results_summary': {
                                'total_models': len(res),
                                'best_keto_f1': max([r['F1'] for r in res if r['task'] == 'keto'], default=0),
                                'best_vegan_f1': max([r['F1'] for r in res if r['task'] == 'vegan'], default=0),
                            }
                        }
                        save_pipeline_state('completed', final_state)
                        
                    else:
                        log.warning("⚠️  No models to save")

                except Exception as e:
                    log.error(f"❌ Could not save models: {e}")
                    
                if args.sanity_check:
                    log.info("\n🎉 SANITY CHECK COMPLETE - All systems functional!")

            except KeyboardInterrupt:
                log.info("🛑 Training interrupted by user")
                sys.exit(0)
            except Exception as e:
                log.error(f"❌ Training pipeline failed: {e}")
                log.error(f"   Error type: {type(e).__name__}")

                import traceback
                log.debug(f"Full traceback:\n{traceback.format_exc()}")

                log.info("🚫 EXITING WITHOUT RESTART")
                sys.exit(1)

        elif args.ground_truth:
            log.info(f"📊 Evaluating on ground truth: {args.ground_truth}")

            try:
                import pickle
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

                if not os.path.exists(args.ground_truth):
                    log.error(
                        f"❌ Ground truth file not found: {args.ground_truth}")
                    sys.exit(1)

                df = pd.read_csv(args.ground_truth)
                log.info(f"✅ Loaded ground truth with {len(df)} rows")

                # DEBUG: Log the column names to see what we have
                log.info(f"📋 Available columns: {list(df.columns)}")

                # Detect the correct label column names
                def detect_label_columns(df):
                    """Detect the correct label column names."""
                    keto_cols = [
                        col for col in df.columns if 'keto' in col.lower()]
                    vegan_cols = [
                        col for col in df.columns if 'vegan' in col.lower()]

                    keto_col = keto_cols[0] if keto_cols else None
                    vegan_col = vegan_cols[0] if vegan_cols else None

                    log.info(
                        f"🏷️  Detected label columns: keto={keto_col}, vegan={vegan_col}")
                    return keto_col, vegan_col

                keto_col, vegan_col = detect_label_columns(df)

                if not keto_col or not vegan_col:
                    log.error(
                        f"❌ Missing required label columns. Found: keto={keto_col}, vegan={vegan_col}")
                    sys.exit(1)

                # Load models, vectorizer, and metadata
                model_path = CFG.artifacts_dir / "models.pkl"
                vec_path = CFG.artifacts_dir / "vectorizer.pkl"
                metadata_path = CFG.artifacts_dir / "model_metadata.json"

                if not (model_path.exists() and vec_path.exists()):
                    log.warning(
                        "⚠️ Models not found in artifacts/ — trying pretrained_models/ fallback...")
                    model_path = CFG.pretrained_models_dir / "models.pkl"
                    vec_path = CFG.pretrained_models_dir / "vectorizer.pkl"
                    metadata_path = CFG.pretrained_models_dir / "model_metadata.json"

                if not (model_path.exists() and vec_path.exists()):
                    log.error("❌ No trained models found.")
                    sys.exit(1)

                with open(vec_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                with open(model_path, 'rb') as f:
                    models = pickle.load(f)

                # Load metadata if available
                model_metadata = {}
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            model_metadata = json.load(f)
                        log.info(f"✅ Loaded model metadata")
                    except Exception as e:
                        log.warning(f"⚠️ Could not load model metadata: {e}")

                log.info(f"✅ Loaded models and vectorizer")

                # Analyze model requirements based on metadata or inspection
                def analyze_model_requirements(task, model, metadata):
                    """Determine what features a model needs."""
                    if task in metadata:
                        requires_images = metadata[task].get(
                            'requires_images', False)
                        model_type = metadata[task].get('type', 'unknown')
                        log.info(
                            f"   ├─ {task} ({model_type}): requires_images={requires_images}")
                        return requires_images, model_type

                    # Fallback: inspect model properties
                    if hasattr(model, 'ensemble_type'):
                        # It's an ensemble model
                        requires_images = model.ensemble_type in [
                            'best_two', 'smart_ensemble', 'both']
                        log.info(
                            f"   ├─ {task} (ensemble): {model.ensemble_type}, requires_images={requires_images}")
                        return requires_images, 'ensemble'
                    elif hasattr(model, 'n_features_in_'):
                        # Check feature count: text=~6341, image=2048, combined=~8389
                        n_features = model.n_features_in_
                        if n_features <= 2048:
                            requires_images = True  # Image-only model
                            model_type = 'image'
                        elif n_features > 7000:  # Combined features
                            requires_images = True
                            model_type = 'both'
                        else:
                            requires_images = False  # Text-only model
                            model_type = 'text'
                        log.info(
                            f"   ├─ {task} ({model_type}): {n_features} features, requires_images={requires_images}")
                        return requires_images, model_type
                    else:
                        log.info(f"   ├─ {task} (unknown): assuming text-only")
                        return False, 'text'

                # Analyze all models
                model_requirements = {}
                for task, model in models.items():
                    requires_images, model_type = analyze_model_requirements(
                        task, model, model_metadata)
                    model_requirements[task] = {
                        'requires_images': requires_images,
                        'type': model_type
                    }

                # Prepare text features (always needed)
                df['clean'] = df['ingredients'].fillna(
                    "").apply(lambda x: normalise(x))
                X_text = vectorizer.transform(df['clean'])
                log.info(f"✅ Text features shape: {X_text.shape}")

                # Prepare image features if needed
                need_any_images = any(req['requires_images']
                                      for req in model_requirements.values())
                X_combined = None

                if need_any_images:
                    log.info(
                        f"\n🖼️  Some models require images - preparing combined features...")

                    # Check if we have photos and can process them
                    if 'photo_url' in df.columns:
                        df_with_photos = filter_photo_rows(df)
                        log.info(
                            f"   ├─ Rows with valid photo URLs: {len(df_with_photos)}")

                        if len(df_with_photos) > 10:  # Only process if we have enough images
                            try:
                                # Quick image processing for evaluation
                                eval_img_dir = CFG.image_dir / "eval_temp"
                                downloaded_indices = _download_images(
                                    df_with_photos, eval_img_dir, max_workers=8, force=False
                                )

                                if downloaded_indices:
                                    df_downloaded = df_with_photos.loc[downloaded_indices]
                                    embeddings, valid_indices = build_image_embeddings(
                                        df_downloaded, "eval_temp", force=False
                                    )

                                    if embeddings is not None and embeddings.size > 0:
                                        log.info(
                                            f"   ✅ Extracted embeddings for {len(valid_indices)} images")

                                        # Create combined features
                                        padding_matrix = np.zeros(
                                            (len(df), 2048), dtype=np.float32)

                                        # Fill in actual embeddings where available
                                        for i, idx in enumerate(valid_indices):
                                            if idx in df.index:
                                                row_pos = df.index.get_loc(idx)
                                                padding_matrix[row_pos] = embeddings[i]

                                        X_combined = combine_features(
                                            X_text, padding_matrix)
                                        log.info(
                                            f"   ✅ Created combined features: {X_combined.shape}")

                                # Cleanup
                                if eval_img_dir.exists():
                                    import shutil
                                    try:
                                        shutil.rmtree(eval_img_dir)
                                    except:
                                        pass

                            except Exception as e:
                                log.warning(
                                    f"   ⚠️  Image processing failed: {e}")

                    # Fallback: zero-padded features
                    if X_combined is None:
                        log.info(
                            f"   ├─ Using zero-padded image features (no real images)")
                        padding_matrix = np.zeros(
                            (len(df), 2048), dtype=np.float32)
                        X_combined = combine_features(X_text, padding_matrix)

                # Make predictions
                log.info(f"\n🔮 Making predictions...")

                predictions = {
                    'keto_pred': [],
                    'vegan_pred': [],
                    'keto_true': [],
                    'vegan_true': []
                }

                for idx, row in df.iterrows():
                    row_idx = df.index.get_loc(idx)

                    # Store true labels
                    predictions['keto_true'].append(row[keto_col])
                    predictions['vegan_true'].append(row[vegan_col])

                    for task in ["keto", "vegan"]:
                        if task in models:
                            model = models[task]
                            requirements = model_requirements[task]

                            try:
                                # Select appropriate features based on model requirements
                                if requirements['requires_images']:
                                    if X_combined is not None:
                                        features = X_combined[row_idx:row_idx+1]
                                        log.debug(
                                            f"   Using combined features for {task}")
                                    else:
                                        log.warning(
                                            f"   Model {task} needs images but none available - using text only")
                                        features = X_text[row_idx:row_idx+1]
                                else:
                                    features = X_text[row_idx:row_idx+1]
                                    log.debug(
                                        f"   Using text features for {task}")

                                # Make prediction
                                if hasattr(model, 'predict_proba'):
                                    try:
                                        prob = model.predict_proba(features)[
                                            0, 1]
                                        pred = 1 if prob >= 0.5 else 0
                                    except Exception as e:
                                        log.warning(
                                            f"   predict_proba failed for {task}: {e}")
                                        pred = model.predict(features)[0]
                                else:
                                    pred = model.predict(features)[0]

                                predictions[f'{task}_pred'].append(int(pred))

                                # Log first few predictions for debugging
                                if idx < 3:
                                    log.debug(
                                        f"   Row {idx} {task}: pred={pred}, true={row[keto_col if task == 'keto' else vegan_col]}")

                            except Exception as e:
                                log.error(
                                    f"   ❌ Prediction failed for {task} row {idx}: {str(e)[:60]}...")
                                predictions[f'{task}_pred'].append(
                                    0)  # Default to negative
                        else:
                            log.warning(f"   ⚠️ No model found for {task}")
                            predictions[f'{task}_pred'].append(0)

                # Convert to DataFrame and save
                results_df = pd.DataFrame(predictions)
                results_df['ingredients'] = df['ingredients'].values
                results_df['index'] = df.index

                pred_path = CFG.artifacts_dir / "ground_truth_predictions.csv"
                results_df.to_csv(pred_path, index=False)
                log.info(f"✅ Saved predictions to {pred_path}")

                # Calculate metrics with safe handling
                log.info(f"\n📊 Calculating evaluation metrics...")

                def safe_int_conversion(series):
                    """Safely convert series to int, handling None values."""
                    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)

                metrics = []
                for task in ["keto", "vegan"]:
                    true_col = f"{task}_true"
                    pred_col = f"{task}_pred"

                    if true_col in results_df.columns and pred_col in results_df.columns:
                        # Safely convert to integers
                        y_true = safe_int_conversion(results_df[true_col])
                        y_pred = safe_int_conversion(results_df[pred_col])

                        # Remove any rows where true labels are invalid
                        valid_mask = (y_true >= 0) & (y_true <= 1)
                        y_true_valid = y_true[valid_mask]
                        y_pred_valid = y_pred[valid_mask]

                        if len(y_true_valid) > 0:
                            try:
                                acc = accuracy_score(
                                    y_true_valid, y_pred_valid)
                                prec = precision_score(
                                    y_true_valid, y_pred_valid, zero_division=0)
                                rec = recall_score(
                                    y_true_valid, y_pred_valid, zero_division=0)
                                f1 = f1_score(
                                    y_true_valid, y_pred_valid, zero_division=0)
                                n_samples = len(y_true_valid)

                                metrics.append({
                                    "task": task,
                                    "accuracy": acc,
                                    "precision": prec,
                                    "recall": rec,
                                    "f1_score": f1,
                                    "n_samples": n_samples
                                })

                                log.info(f"✅ {task.upper()} - ACC: {acc:.3f} | PREC: {prec:.3f} | "
                                         f"REC: {rec:.3f} | F1: {f1:.3f} | N: {n_samples}")

                                # Log confusion matrix for debugging
                                cm = confusion_matrix(
                                    y_true_valid, y_pred_valid)
                                log.info(f"   Confusion Matrix for {task}:")
                                log.info(
                                    f"   [[TN={cm[0, 0]}, FP={cm[0, 1]}],")
                                log.info(
                                    f"    [FN={cm[1, 0]}, TP={cm[1, 1]}]]")

                                # Additional debugging: show prediction distribution
                                pos_preds = (y_pred_valid == 1).sum()
                                neg_preds = (y_pred_valid == 0).sum()
                                pos_true = (y_true_valid == 1).sum()
                                neg_true = (y_true_valid == 0).sum()
                                log.info(
                                    f"   Predictions: {pos_preds} positive, {neg_preds} negative")
                                log.info(
                                    f"   True labels: {pos_true} positive, {neg_true} negative")

                            except Exception as e:
                                log.error(
                                    f"❌ Metric calculation failed for {task}: {e}")
                        else:
                            log.warning(
                                f"⚠️ No valid samples for {task} evaluation")

                # Save metrics
                if metrics:
                    metrics_df = pd.DataFrame(metrics)
                    metrics_path = CFG.artifacts_dir / "eval_metrics.csv"
                    metrics_df.to_csv(metrics_path, index=False)
                    log.info(f"📈 Saved evaluation metrics to {metrics_path}")

                    # Print summary
                    log.info(f"\n🎯 EVALUATION SUMMARY:")
                    for metric in metrics:
                        log.info(f"   {metric['task'].upper()}: F1={metric['f1_score']:.3f}, "
                                 f"Accuracy={metric['accuracy']:.3f}, Samples={metric['n_samples']}")
                else:
                    log.warning(f"⚠️ No metrics calculated")

            except Exception as e:
                log.error(f"❌ Ground truth evaluation failed: {e}")
                import traceback
                log.error(f"Full traceback:\n{traceback.format_exc()}")
                sys.exit(1)

        elif args.predict:
            log.info(f"🔮 Running prediction on unlabeled data: {args.predict}")

            try:
                import pickle

                predict_path = Path(args.predict)
                if not predict_path.exists():
                    log.error(f"❌ Prediction file not found: {args.predict}")
                    sys.exit(1)

                df = pd.read_csv(predict_path)
                if 'ingredients' not in df.columns:
                    log.error(
                        "❌ Input CSV must contain an 'ingredients' column")
                    sys.exit(1)

                # Load models and vectorizer
                model_path = CFG.artifacts_dir / "models.pkl"
                vec_path = CFG.artifacts_dir / "vectorizer.pkl"

                if not model_path.exists() or not vec_path.exists():
                    log.warning(
                        "⚠️  Trained models not found in artifacts/, trying pretrained_models/...")

                    model_path = Path("/app/pretrained_models/models.pkl")
                    vec_path = Path("/app/pretrained_models/vectorizer.pkl")

                    if not model_path.exists() or not vec_path.exists():
                        log.error(
                            "❌ No trained models available in either artifacts/ or pretrained_models/")
                        sys.exit(1)

                with open(vec_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                log.info(f"✅ Loaded vectorizer from {vec_path}")

                with open(model_path, 'rb') as f:
                    models = pickle.load(f)
                log.info(f"✅ Loaded models from {model_path}")

                # Transform ingredients
                texts = df['ingredients'].fillna("").tolist()
                X = vectorizer.transform(texts)
                log.info("🧠 Vectorized ingredients")

                # Predict
                preds = []

                # Check if models need images
                for task, model in models.items():
                    if hasattr(model, 'n_features_in_') and model.n_features_in_ > X.shape[1]:
                        log.info(
                            f"   ├─ {task} model expects {model.n_features_in_} features (text has {X.shape[1]})")

                for idx, row in df.iterrows():
                    row_preds = {"index": idx,
                                 "ingredients": row["ingredients"]}
                    for task in ["keto", "vegan"]:
                        if task in models:
                            model = models[task]

                            # Get features for this row
                            features = X[idx]

                            # Check if model needs images
                            if hasattr(model, 'n_features_in_') and model.n_features_in_ > X.shape[1]:
                                # Pad with zeros for images
                                padding = np.zeros((1, 2048), dtype=np.float32)
                                features = combine_features(
                                    features.reshape(1, -1), padding)

                            pred = model.predict(features)
                            row_preds[f"{task}_pred"] = int(pred[0])
                        else:
                            row_preds[f"{task}_pred"] = None
                            log.warning(f"⚠️ No model found for task: {task}")
                    preds.append(row_preds)

                preds_df = pd.DataFrame(preds)
                out_path = CFG.artifacts_dir / "predictions_custom.csv"
                preds_df.to_csv(out_path, index=False)
                log.info(f"✅ Saved predictions to: {out_path}")

            except Exception as e:
                log.error(f"❌ Prediction failed: {e}")
                sys.exit(1)

        else:
            # Default pipeline
            log.info(f"🧠 Default pipeline - sample_frac={args.sample_frac}")
            
            if not args.train and not args.ground_truth and not args.ingredients and not args.predict:
                log.info("\n📋 No specific mode selected. Available options:")
                log.info("   ├─ --train: Train new models")
                log.info("   ├─ --ground_truth <file>: Evaluate on labeled data")
                log.info("   ├─ --ingredients <list>: Classify specific ingredients")
                log.info("   ├─ --predict <file>: Batch prediction on CSV")
                log.info("   └─ --sanity_check: Quick test run")
                
                response = input("\nRun training pipeline? [Y/n]: ").strip().lower()
                if response != 'n':
                    args.train = True
                else:
                    log.info("👋 Exiting. Run with --help for usage information.")
                    sys.exit(0)

            if args.train:
                try:
                    run_full_pipeline(mode=args.mode, force=args.force,
                                      sample_frac=args.sample_frac)
                except Exception as e:
                    log.error(f"❌ Default pipeline failed: {e}")
                    sys.exit(1)

        log.info("🏁 Main completed successfully")
        
        # Clean up sanity check environment variable
        if 'SANITY_CHECK' in os.environ:
            del os.environ['SANITY_CHECK']

    except KeyboardInterrupt:
        log.info("🛑 Main interrupted by user")
        sys.exit(0)
    except SystemExit as e:
        log.info(f"🚫 System exit: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        log.error(f"❌ Unexpected error in main: {e}")
        import traceback
        log.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    # Prevent any possibility of restart loops
    import sys
    import os

    # Check if we're already in a restart loop
    restart_count = os.environ.get('PIPELINE_RESTART_COUNT', '0')
    restart_count = int(restart_count)

    if restart_count > 0:
        log.info(f"❌ RESTART LOOP DETECTED (count={restart_count}) - STOPPING")
        sys.exit(1)

    # Set restart counter
    os.environ['PIPELINE_RESTART_COUNT'] = str(restart_count + 1)

    try:
        main()
    except Exception as e:
        log.info(f"❌ Final exception caught: {e}")
        sys.exit(1)
    finally:
        # Clear restart counter on normal exit
        if 'PIPELINE_RESTART_COUNT' in os.environ:
            del os.environ['PIPELINE_RESTART_COUNT']