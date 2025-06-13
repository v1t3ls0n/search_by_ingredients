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
    • USDA-based carbohydrate filtering (≤10g carbs/100g → keto-safe)
    • Phrase-level disqualifications (e.g., "chicken broth")
    • Whitelist override of verified-safe ingredients (e.g., "almond flour")
    • Soft ML fallback + rule-based priority merging
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
   - Uses USDA nutritional DB for rule-based classification
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

Robust against partial data, broken images, or failed downloads.
Supports interactive development, Docker builds, and production use.

Author: Guy Vitelson (aka @v1t3ls0n on GitHub)
"""


# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
"""
This section imports all required libraries, organizing them by category:
- Future compatibility imports
- Standard library modules
- Core third-party libraries (NumPy, Pandas, etc.)
- Machine learning libraries (scikit-learn, LightGBM)
- Deep learning libraries (PyTorch, torchvision)
- Specialized libraries (NLTK, imbalanced-learn)
"""

# --- Future compatibility ---
from __future__ import annotations

# --- Standard library ---
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
from pathlib import Path

# --- Third-party: core ---
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
import psutil

# --- NLTK (used for lemmatization) ---
import nltk
from nltk.stem import WordNetLemmatizer
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

# --- Imbalanced learning ---
from imblearn.over_sampling import SMOTE, RandomOverSampler

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
    Immutable configuration container for the pipeline.
    
    Attributes:
        artifacts_dir: Directory for saving trained models and vectorizers
        data_dir: Root directory for data files
        usda_dir: Directory containing USDA nutritional database files
        url_map: Mapping of dataset names to file paths/URLs
        vec_kwargs: Parameters for TF-IDF vectorizer
        image_dir: Directory for storing downloaded recipe images
    """
    pretrained_models_dir: Path = Path("/app/pretrained_models")
    artifacts_dir: Path = Path("/app/artifacts")
    data_dir: Path = Path("/app/data")
    usda_dir: Path  = Path("/app/data/usda")
    url_map: Mapping[str, str] = field(default_factory=lambda: {
        "allrecipes.parquet": "/app/data/allrecipes.parquet",
        "ground_truth_sample.csv": "/app/data/ground_truth_sample.csv",
    })
    vec_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        min_df=2, ngram_range=(1, 3), max_features=50000, sublinear_tf=True))
    image_dir: Path = Path("dataset/arg_max/images")

CFG = Config()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
"""
Sets up comprehensive logging with timestamps and severity levels.
All major operations are logged for debugging and monitoring.
"""
# Make sure artifacts dir exists (harmless if already exists via Docker)
CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)


# Define log file path
log_file = CFG.artifacts_dir / "pipeline.log"


# Set up logger
log = logging.getLogger("PIPE")
log.setLevel(logging.INFO)

# Define formatter
formatter = logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s", datefmt="%H:%M:%S")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
log.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)


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


def _download_images(df: pd.DataFrame, img_dir: Path, max_workers: int = 16) -> list[int]:
    """
    Download images using multithreading with comprehensive logging and progress tracking.

    This function downloads recipe images from URLs specified in the DataFrame,
    implementing robust error handling, retry mechanisms, and detailed progress tracking.
    It's designed to handle large-scale downloads efficiently while providing
    comprehensive feedback on the download process.

    Enhanced Features:
    - Real-time download statistics and bandwidth monitoring
    - URL validation and preprocessing
    - Error categorization and analysis
    - Retry mechanisms for failed downloads
    - File integrity verification
    - Memory-efficient processing
    - Detailed error logging

    Args:
        df: DataFrame containing photo_url column with image URLs
        img_dir: Directory to save downloaded images
        max_workers: Maximum number of concurrent download threads

    Returns:
        List of successful indices for filtering downstream processing

    Example:
        >>> recipes_df = pd.DataFrame({'photo_url': ['http://example.com/img1.jpg']})
        >>> valid_indices = _download_images(recipes_df, Path('./images'), max_workers=8)
        >>> print(f"Downloaded {len(valid_indices)} images")
    """
    import time
    import hashlib
    from collections import defaultdict, Counter
    from urllib.parse import urlparse
    import threading

    download_start = time.time()

    # ------------------------------------------------------------------
    # Initialization and Validation
    # ------------------------------------------------------------------
    log.info(f"\n📥 IMAGE DOWNLOAD PIPELINE: {img_dir.name}")
    log.info(f"   Target directory: {img_dir}")
    log.info(f"   Max workers: {max_workers}")
    log.info(f"   Total URLs to process: {len(df):,}")

    # Check if PyTorch/PIL available for image processing
    if not TORCH_AVAILABLE:
        log.warning("   ⚠️  PyTorch not available - skipping image downloads")
        return []

    # Backup-based early exit
    backup_emb = img_dir / "embeddings.npy"
    if backup_emb.exists():
        try:
            num_jpgs = len(list(img_dir.glob("*.jpg")))
            if num_jpgs >= len(df):
                log.info(
                    f"   📦 Backup detected: {num_jpgs} images + existing embeddings → skipping downloads")
                return sorted(df.index.tolist())  # Return all indices as valid
        except Exception as e:
            log.warning(f"   ⚠️ Could not verify backup completeness: {e}")

    # Create directory structure
    img_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"   ✅ Directory created/verified: {img_dir}")

    # Validate DataFrame structure
    if 'photo_url' not in df.columns:
        log.error("   ❌ No 'photo_url' column found in DataFrame")
        return []

    # ------------------------------------------------------------------
    # URL Analysis and Preprocessing
    # ------------------------------------------------------------------
    log.info(f"\n   🔍 URL Analysis:")

    with tqdm(total=3, desc="      ├─ Analyzing URLs", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as analysis_pbar:

        analysis_pbar.set_description("      ├─ Filtering valid URLs")

        # Analyze URL patterns and validity
        url_analysis = {
            'total': len(df),
            'valid_http': 0,
            'already_exists': 0,
            'invalid_format': 0,
            'empty_null': 0,
            'domains': Counter(),
            'extensions': Counter()
        }

        valid_downloads = []
        skipped_existing = []

        for idx, url in df['photo_url'].items():
            img_path = img_dir / f"{idx}.jpg"

            # Check if already exists
            if img_path.exists():
                url_analysis['already_exists'] += 1
                skipped_existing.append(idx)
                continue

            # Validate URL format
            if not isinstance(url, str) or not url.strip():
                url_analysis['empty_null'] += 1
                continue

            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url_analysis['invalid_format'] += 1
                continue

            # URL is valid for download
            url_analysis['valid_http'] += 1
            valid_downloads.append((idx, url))

            # Analyze domain and extension
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                url_analysis['domains'][domain] += 1

                # Extract file extension from path
                path_parts = parsed.path.lower().split('.')
                if len(path_parts) > 1:
                    ext = path_parts[-1][:10]  # Limit length
                    url_analysis['extensions'][ext] += 1
            except Exception:
                pass

        analysis_pbar.update(1)

        analysis_pbar.set_description("      ├─ Generating statistics")

        # Log URL analysis results
        log.info(f"      📊 URL Statistics:")
        log.info(f"      ├─ Total URLs: {url_analysis['total']:,}")
        log.info(
            f"      ├─ Valid HTTP(S): {url_analysis['valid_http']:,} ({url_analysis['valid_http']/url_analysis['total']*100:.1f}%)")
        log.info(f"      ├─ Already exist: {url_analysis['already_exists']:,}")
        log.info(
            f"      ├─ Invalid format: {url_analysis['invalid_format']:,}")
        log.info(f"      └─ Empty/null: {url_analysis['empty_null']:,}")

        analysis_pbar.update(1)

        # Show top domains
        if url_analysis['domains']:
            top_domains = url_analysis['domains'].most_common(5)
            log.info(f"      📊 Top Domains:")
            for domain, count in top_domains:
                log.info(
                    f"      ├─ {domain}: {count:,} images ({count/url_analysis['valid_http']*100:.1f}%)")

        # Show file extensions
        if url_analysis['extensions']:
            top_extensions = url_analysis['extensions'].most_common(3)
            log.info(f"      📊 File Extensions:")
            for ext, count in top_extensions:
                log.info(f"      ├─ .{ext}: {count:,}")

        analysis_pbar.update(1)

    # Early exit if no downloads needed
    if not valid_downloads:
        log.info(
            f"   ✅ No new downloads needed (all {url_analysis['already_exists']} images exist)")
        return skipped_existing

    log.info(
        f"   🎯 Download Plan: {len(valid_downloads):,} new images to download")

    # ------------------------------------------------------------------
    # Download Execution with Enhanced Tracking
    # ------------------------------------------------------------------
    log.info(f"\n   🚀 Starting parallel downloads...")

    # Shared statistics with thread safety
    stats_lock = threading.Lock()
    download_stats = {
        "downloaded": 0,
        "exists": 0,
        "invalid": 0,
        "failed": 0,
        "bytes_downloaded": 0,
        "retry_success": 0
    }

    valid_indices = list(skipped_existing)  # Include pre-existing
    failed_urls = []
    download_times = []

    def fetch_with_retry(idx_url, max_retries=2):
        """
        Enhanced fetch function with retry logic and detailed error handling.
        
        This internal function handles individual image downloads with:
        - Automatic retry on failure
        - Content validation
        - Atomic file writing
        - Detailed error tracking
        
        Args:
            idx_url: Tuple of (index, url) to download
            max_retries: Number of retry attempts
            
        Returns:
            Tuple of (status, index, url, error, size, [fetch_time])
        """
        idx, url = idx_url
        img_path = img_dir / f"{idx}.jpg"

        # Double-check existence (race condition safety)
        if img_path.exists():
            with stats_lock:
                stats_check = os.path.getsize(img_path)
                download_stats["exists"] += 1
                if stats_check > 0:
                    return "exists", idx, url, None, stats_check

        # Validate URL format (redundant check for thread safety)
        if not isinstance(url, str) or not url.strip().startswith("http"):
            with stats_lock:
                download_stats["invalid"] += 1
            return "invalid", idx, url, "Invalid URL format", 0

        # Attempt download with retries
        last_error = None
        for attempt in range(max_retries + 1):
            fetch_start = time.time()

            try:
                # Configure request with better error handling
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; RecipeImageDownloader/1.0)',
                    'Accept': 'image/*,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }

                resp = requests.get(
                    url,
                    timeout=15,  # Increased timeout
                    headers=headers,
                    allow_redirects=True,
                    stream=True  # For large images
                )
                resp.raise_for_status()

                # Check content type
                content_type = resp.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif']):
                    raise ValueError(f"Invalid content type: {content_type}")

                # Download with size validation
                content = resp.content
                if len(content) < 100:  # Minimum viable image size
                    raise ValueError(f"Image too small: {len(content)} bytes")

                if len(content) > 50 * 1024 * 1024:  # 50MB limit
                    raise ValueError(f"Image too large: {len(content)} bytes")

                # Write file atomically
                temp_path = img_path.with_suffix('.tmp')
                with open(temp_path, 'wb') as fh:
                    fh.write(content)

                # Verify file integrity
                if os.path.getsize(temp_path) != len(content):
                    os.remove(temp_path)
                    raise ValueError("File size mismatch after write")

                # Atomic move to final location
                temp_path.rename(img_path)

                # Update statistics
                fetch_time = time.time() - fetch_start
                with stats_lock:
                    if attempt > 0:
                        download_stats["retry_success"] += 1
                    download_stats["downloaded"] += 1
                    download_stats["bytes_downloaded"] += len(content)

                return "downloaded", idx, url, None, len(content), fetch_time

            except requests.exceptions.Timeout:
                last_error = f"Timeout after 15s (attempt {attempt+1})"
            except requests.exceptions.ConnectionError:
                last_error = f"Connection error (attempt {attempt+1})"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code} (attempt {attempt+1})"
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)[:50]} (attempt {attempt+1})"
            except ValueError as e:
                last_error = f"Validation error: {str(e)} (attempt {attempt+1})"
                break  # Don't retry validation errors
            except Exception as e:
                last_error = f"Unexpected error: {str(e)[:50]} (attempt {attempt+1})"

            # Brief pause before retry
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # All attempts failed
        with stats_lock:
            download_stats["failed"] += 1

        return "failed", idx, url, last_error, 0

    # ------------------------------------------------------------------
    # Parallel Download Execution
    # ------------------------------------------------------------------
    bandwidth_samples = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = [executor.submit(fetch_with_retry, idx_url)
                   for idx_url in valid_downloads]

        # Progress bar with real-time statistics
        progress_bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"      ├─ Downloading {img_dir.name}",
            position=1, leave=False,
            bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}] {rate_fmt}"
        )

        completed = 0
        for future in progress_bar:
            completed += 1
            result = future.result()

            # Unpack result (handle variable return length)
            if len(result) >= 5:
                status, idx, url, error, size = result[:5]
                fetch_time = result[5] if len(result) > 5 else 0
            else:
                status, idx, url, error, size = result + \
                    (0,) * (5 - len(result))
                fetch_time = 0

            # Track successful downloads
            if status in {"downloaded", "exists"}:
                valid_indices.append(idx)

                # Calculate bandwidth for downloaded files
                if status == "downloaded" and fetch_time > 0 and size > 0:
                    bandwidth_mbps = (size / (1024 * 1024)) / fetch_time
                    bandwidth_samples.append(bandwidth_mbps)
                    download_times.append(fetch_time)

            elif status == "failed":
                failed_urls.append((idx, url, error))

            # Update progress bar with live statistics
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed > 0:
                downloads_per_sec = completed / elapsed

                # Calculate current stats safely
                with stats_lock:
                    current_stats = download_stats.copy()

                # Estimate completion
                remaining = len(futures) - completed
                eta_seconds = remaining / downloads_per_sec if downloads_per_sec > 0 else 0

                # Update progress description with live stats
                progress_bar.set_postfix({
                    'Success': f"{current_stats['downloaded'] + current_stats['exists']}",
                    'Failed': f"{current_stats['failed']}",
                    'Speed': f"{downloads_per_sec:.1f}/s",
                    'ETA': f"{eta_seconds:.0f}s" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
                })

    # ------------------------------------------------------------------
    # Download Results Analysis
    # ------------------------------------------------------------------
    total_time = time.time() - download_start

    log.info(f"\n   📊 DOWNLOAD RESULTS:")
    log.info(f"   ├─ Total processing time: {total_time:.1f}s")
    log.info(
        f"   ├─ Successfully downloaded: {download_stats['downloaded']:,}")
    log.info(f"   ├─ Already existed: {download_stats['exists']:,}")
    log.info(f"   ├─ Failed downloads: {download_stats['failed']:,}")
    log.info(f"   ├─ Invalid URLs: {download_stats['invalid']:,}")
    log.info(f"   └─ Retry successes: {download_stats['retry_success']:,}")

    # Success rate analysis
    total_attempted = download_stats['downloaded'] + download_stats['failed']
    if total_attempted > 0:
        success_rate = download_stats['downloaded'] / total_attempted * 100
        log.info(
            f"   📈 Success rate: {success_rate:.1f}% ({download_stats['downloaded']}/{total_attempted})")

    # Performance metrics
    if download_stats['downloaded'] > 0:
        log.info(f"\n   ⚡ Performance Metrics:")
        log.info(
            f"   ├─ Download speed: {download_stats['downloaded']/total_time:.1f} images/second")
        log.info(
            f"   ├─ Data downloaded: {download_stats['bytes_downloaded']/(1024*1024):.1f} MB")

        if bandwidth_samples:
            avg_bandwidth = sum(bandwidth_samples) / len(bandwidth_samples)
            max_bandwidth = max(bandwidth_samples)
            log.info(f"   ├─ Average bandwidth: {avg_bandwidth:.1f} MB/s")
            log.info(f"   └─ Peak bandwidth: {max_bandwidth:.1f} MB/s")

        if download_times:
            avg_time = sum(download_times) / len(download_times)
            log.info(f"   └─ Average download time: {avg_time:.2f}s per image")

    # ------------------------------------------------------------------
    # Error Analysis and Logging
    # ------------------------------------------------------------------
    if failed_urls:
        log.info(f"\n   ⚠️  Error Analysis ({len(failed_urls)} failures):")

        # Categorize errors
        error_categories = defaultdict(int)
        error_examples = defaultdict(list)

        for idx, url, error in failed_urls:
            # Categorize error types
            if not error:
                category = "Unknown"
            elif "timeout" in error.lower():
                category = "Timeout"
            elif "connection" in error.lower():
                category = "Connection"
            elif "404" in error or "not found" in error.lower():
                category = "Not Found"
            elif "403" in error or "forbidden" in error.lower():
                category = "Forbidden"
            elif "validation" in error.lower():
                category = "Invalid Content"
            else:
                category = "Other"

            error_categories[category] += 1
            # Keep max 3 examples per category
            if len(error_examples[category]) < 3:
                error_examples[category].append(
                    (idx, url[:50] + "..." if len(url) > 50 else url, error))

        # Log error summary
        for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(failed_urls) * 100
            log.info(f"   ├─ {category}: {count} ({percentage:.1f}%)")

            # Show examples for major error categories
            if count >= 5 and error_examples[category]:
                for idx, url_short, error in error_examples[category][:2]:
                    log.info(
                        f"   │  └─ Example: {url_short} - {error[:60]}...")

        # Save detailed error log
        fail_log_path = img_dir / "failed_downloads.txt"
        try:
            with open(fail_log_path, "w", encoding='utf-8') as f:
                f.write("Index\tURL\tError\tTimestamp\n")
                for idx, url, error in failed_urls:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{idx}\t{url}\t{error}\t{timestamp}\n")

            log.info(f"   💾 Detailed error log saved to: {fail_log_path}")

        except Exception as e:
            log.warning(f"   ⚠️  Failed to save error log: {e}")

    # ------------------------------------------------------------------
    # Disk Usage and Cleanup
    # ------------------------------------------------------------------
    try:
        # Calculate disk usage
        total_size = 0
        image_count = 0
        for img_file in img_dir.glob("*.jpg"):
            if img_file.is_file():
                total_size += img_file.stat().st_size
                image_count += 1

        if image_count > 0:
            log.info(f"\n   💾 Storage Summary:")
            log.info(f"   ├─ Images stored: {image_count:,}")
            log.info(f"   ├─ Total size: {total_size/(1024*1024):.1f} MB")
            log.info(
                f"   └─ Average size: {total_size/(1024*1024)/image_count:.2f} MB per image")

    except Exception as e:
        log.warning(f"   ⚠️  Storage analysis failed: {e}")

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    total_valid = len(valid_indices)
    log.info(f"\n   🏁 DOWNLOAD COMPLETE:")
    log.info(
        f"   ├─ Total valid images: {total_valid:,}/{len(df):,} ({total_valid/len(df)*100:.1f}%)")
    log.info(f"   ├─ Processing rate: {len(df)/total_time:.1f} URLs/second")
    log.info(f"   ├─ Thread efficiency: {max_workers} workers")
    log.info(f"   └─ Directory: {img_dir}")

    # Memory cleanup for large operations
    if len(df) > 10000:
        import gc
        gc.collect()
        log.debug(f"   🧹 Memory cleanup completed")

    return valid_indices


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
    import time
    import requests
    from urllib.parse import urlparse
    import warnings
    from collections import Counter

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
    import gc
    gc.collect()

    return silver, ground_truth, recipes, carb_df


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


# =============================================================================
# SANITY CHECKS
# =============================================================================
"""
Basic tests to ensure the classification system is working correctly.
These checks run on module import to catch configuration errors early.
"""

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
            from . import run_full_pipeline, BEST
            
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
# MEMORY OPTIMIZATION
# =============================================================================
"""
Functions for managing memory usage during training. Critical for handling
large datasets and preventing out-of-memory errors, especially when working
with both text and image features.
"""

def optimize_memory_usage(stage_name=""):
    """
    Optimize memory usage during training with detailed logging.
    
    This function performs garbage collection and clears GPU memory if available.
    It's called between major pipeline stages to prevent memory accumulation.
    
    Args:
        stage_name: Optional name of the current stage for logging
        
    Returns:
        String indicating memory status: "normal", "moderate", "high", or "error"
        
    Example:
        >>> optimize_memory_usage("After image processing")
        🧹 After image processing: Memory cleanup
        ├─ RAM: 45.2% used (7234 MB)
        ├─ RAM freed: 523.4 MB
        └─ Objects collected: 1247
    """
    import gc
    
    # Get memory before cleanup
    try:
        memory_before = psutil.virtual_memory()
        memory_before_used = memory_before.used
        memory_before_percent = memory_before.percent
    except Exception as e:
        log.error(f"Failed to get initial memory stats: {e}")
        return "error"

    # Force garbage collection
    try:
        collected = gc.collect()
    except Exception as e:
        log.debug(f"Garbage collection failed: {e}")
        collected = 0

    # Clear GPU cache if available
    gpu_freed = 0
    if torch and torch.cuda.is_available():
        try:
            gpu_before = torch.cuda.memory_allocated() / (1024**2)  # MB
            torch.cuda.empty_cache()
            gpu_after = torch.cuda.memory_allocated() / (1024**2)  # MB
            gpu_freed = max(0, gpu_before - gpu_after)
        except Exception as e:
            log.debug(f"GPU memory cleanup failed: {e}")
            gpu_freed = 0

    # Get memory after cleanup
    try:
        memory_after = psutil.virtual_memory()
        memory_after_used = memory_after.used
        memory_after_percent = memory_after.percent

        # Calculate memory freed
        memory_freed_bytes = max(0, memory_before_used - memory_after_used)
        memory_freed_mb = memory_freed_bytes / (1024**2)

    except Exception as e:
        log.error(f"Failed to get final memory stats: {e}")
        return "error"

    # Log results
    stage_prefix = f"{stage_name}: " if stage_name else ""
    log.info(f"   🧹 {stage_prefix}Memory cleanup")
    log.info(
        f"      ├─ RAM: {memory_after_percent:.1f}% used ({memory_after_used // (1024**2)} MB)")

    if memory_freed_mb > 1.0:  # Only log if significant
        log.info(f"      ├─ RAM freed: {memory_freed_mb:.1f} MB")

    if collected > 0:
        log.info(f"      ├─ Objects collected: {collected}")

    if gpu_freed > 1.0:
        log.info(f"      ├─ GPU freed: {gpu_freed:.1f} MB")

    # Return status based on memory usage
    if memory_after_percent > 85:
        log.warning(
            f"      ⚠️  High memory usage: {memory_after_percent:.1f}%")
        return "high"
    elif memory_after_percent > 70:
        log.warning(
            f"      ⚠️  Moderate memory usage: {memory_after_percent:.1f}%")
        return "moderate"
    else:
        log.info(f"      ✅ Memory usage normal: {memory_after_percent:.1f}%")
        return "normal"


def handle_memory_crisis():
    """
    Emergency memory cleanup when usage is critical.
    
    Applies aggressive memory optimization techniques including:
    - Multiple garbage collection passes
    - Complete GPU memory clearing
    - Python cache invalidation
    - Memory compaction
    
    Returns:
        Final memory usage percentage after cleanup
    """
    import gc
    
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
            gc.set_debug(0)  # Disable debugging to save memory
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

        return final_percent

    except Exception as e:
        log.error(f"Memory crisis handling failed: {e}")
        # Fallback: return a safe high value
        try:
            return psutil.virtual_memory().percent
        except:
            return 90.0  # Assume high usage if we can't measure


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
    
    Ensures alignment between the silver dataset and actually available images.
    
    Args:
        silver_df: Silver label DataFrame
        image_dir: Directory containing downloaded images
        
    Returns:
        Filtered DataFrame with only rows that have images
    """
    downloaded_ids = [int(p.stem) for p in (image_dir / "silver").glob("*.jpg")]
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
    print(f"\n── {title} set class counts ─────────────")
    for lab in ("keto", "vegan"):
        for col in (f"label_{lab}", f"silver_{lab}"):
            if col in df.columns:
                tot = len(df)
                if tot == 0:
                    print(f"{lab:>5}: No data available (0 rows)")
                    break
                pos = int(df[col].sum())
                print(f"{lab:>5}: {pos:6}/{tot} ({pos/tot:>5.1%})")
                break


def apply_smote(X, y, max_dense_size: int = int(5e7)):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) for class balancing.
    
    SMOTE creates synthetic examples of the minority class to balance the dataset.
    Falls back to random oversampling for very large sparse matrices.
    
    Args:
        X: Feature matrix (sparse or dense)
        y: Target labels
        max_dense_size: Maximum size for dense conversion
        
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    try:
        counts = np.bincount(y)
        if len(counts) < 2:
            return X, y

        ratio = counts.min() / counts.sum()
        if ratio < 0.4:  # Only apply if minority class < 40%
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

    # SVM pipeline with scaling (works for both domains)
    svm_pipe = make_pipeline(
        MaxAbsScaler(copy=False),          # Preserves sparsity
        SVC(kernel="rbf",
            C=1.0,
            gamma="scale",
            cache_size=2048,               # 2GB kernel cache
            class_weight="balanced",
            tol=1e-2,                      # Looser tolerance for speed
            max_iter=20000,
            random_state=42)
    )

    # Rule-based model (text only)
    if domain in ("text", "both"):
        models["Rule"] = (
            RuleModel("keto", RX_KETO, RX_WL_KETO)
            if task == "keto"
            else RuleModel("vegan", RX_VEGAN, RX_WL_VEGAN)
        )

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
    }

    # Image/mixed-feature models (handle dense features better)
    image_family: Dict[str, BaseEstimator] = {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(512, 128),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    # Add LightGBM if available (excellent for tabular/image features)
    if lgb and domain in ("image", "both"):
        image_family["LGBM"] = lgb.LGBMClassifier(
            num_leaves=63,
            learning_rate=0.1,
            n_estimators=250,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )

    # Assemble model selection based on domain
    if domain == "text":
        models.update(text_family)
    elif domain == "image":
        models.update(image_family)
    elif domain == "both":
        models.update(text_family)
        models.update(image_family)

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
        log.info(f"Adding probability calibration to {estimator.__class__.__name__}")
        try:
            from sklearn.calibration import CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(estimator, cv=3, method='sigmoid')
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
    import time
    from itertools import product

    tune_start = time.time()

    # Check cache
    if name in BEST:
        cached_time = time.time() - tune_start
        log.info(f"            ✅ {name}: Using cached model ({cached_time*1000:.0f}ms)")
        return BEST[name]

    # Get hyperparameter grid
    grid = HYPER.get(name, {})

    if not grid:
        # No hyperparameters to tune - use defaults
        log.info(f"            🔧 {name}: No hyperparameters defined, using defaults")
        
        with tqdm(total=1, desc=f"               ├─ Default Fit",
                  position=4, leave=False,
                  bar_format="               ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as default_pbar:
            BEST[name] = base.fit(X, y)
            default_pbar.update(1)

        fit_time = time.time() - tune_start
        log.info(f"            ✅ {name}: Default fit completed in {fit_time:.1f}s")
        return BEST[name]

    # Calculate total parameter combinations
    param_combinations = 1
    for param_values in grid.values():
        param_combinations *= len(param_values) if isinstance(param_values, list) else 1

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

        log.info(f"            ✅ {name}: Grid search completed in {search_time:.1f}s")
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
                    param_impact = results_df.groupby(param_col)['mean_test_score'].agg(['mean', 'std'])
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
        log.info(f"               ├─ CV Stability: {cv_stability} (std={cv_std:.3f})")

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
            improvement_pct = (improvement / default_score * 100) if default_score > 0 else 0

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

            log.debug(f"               └─ Saved hyperparameters to {hyperparams_file}")

        except Exception as e:
            log.warning(f"               └─ Failed to save hyperparameters: {e}")

    except Exception as e:
        # Error handling with fallback
        search_time = time.time() - tune_start
        log.error(f"            ❌ {name}: Grid search failed after {search_time:.1f}s")
        log.error(f"               └─ Error: {str(e)[:80]}...")

        if log.level <= logging.DEBUG:
            import traceback
            log.debug(f"Full traceback for {name} tuning:\n{traceback.format_exc()}")

        # Fallback to default parameters
        log.info(f"            🛡️  {name}: Falling back to default parameters")

        try:
            with tqdm(total=1, desc=f"               ├─ Fallback Fit",
                      position=4, leave=False,
                      bar_format="               ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as fallback_pbar:
                BEST[name] = base.fit(X, y)
                fallback_pbar.update(1)

            fallback_time = time.time() - tune_start
            log.info(f"            ✅ {name}: Fallback completed in {fallback_time:.1f}s")

        except Exception as fallback_error:
            fallback_time = time.time() - tune_start
            log.error(f"            ❌ {name}: Fallback also failed after {fallback_time:.1f}s")
            log.error(f"               └─ Fallback Error: {str(fallback_error)[:60]}...")
            raise RuntimeError(f"Both grid search and fallback failed for {name}")

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
    import time
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
    X_silver,                 # Feature matrix for silver set
    gold_clean: pd.Series,    # Cleaned ingredient strings (gold)
    X_gold,                   # Feature matrix for gold set
    silver_df: pd.DataFrame,  # Silver DataFrame with labels
    gold_df: pd.DataFrame,    # Gold DataFrame with labels
    *,
    domain: str = "text",     # Feature domain
    apply_smote: bool = True  # Whether to apply SMOTE
) -> list[dict]:
    """
    Train on weak (silver) labels, evaluate on gold standard labels.
    
    This is the main training function that:
    1. Trains multiple models on silver-labeled data
    2. Evaluates them on gold standard test data
    3. Applies rule-based verification to predictions
    4. Returns performance metrics for all models
    
    Args:
        X_silver: Feature matrix for silver training data
        gold_clean: Normalized text for gold data (for rules)
        X_gold: Feature matrix for gold test data
        silver_df: Silver DataFrame with 'silver_keto'/'silver_vegan'
        gold_df: Gold DataFrame with 'label_keto'/'label_vegan'
        domain: 'text', 'image', or 'both'
        apply_smote: Whether to apply SMOTE for class balancing
        
    Returns:
        List of result dictionaries with metrics and predictions
    """
    import time
    from datetime import datetime

    # Initialize results and timing
    results: list[dict] = []
    pipeline_start = time.time()

    # Log pipeline initialization
    log.info("🚀 Starting MODE A Training Pipeline")
    log.info(f"   Domain: {domain}")
    log.info(f"   SMOTE enabled: {apply_smote}")
    log.info(f"   Silver set size: {len(silver_df):,}")
    log.info(f"   Gold set size: {len(gold_df):,}")
    log.info(f"   Feature dimensions: {X_silver.shape}")

    # Show class distribution
    log.info("\n📊 Class Distribution Analysis:")
    for task in ("keto", "vegan"):
        silver_pos = silver_df[f"silver_{task}"].sum()
        silver_total = len(silver_df)
        gold_pos = gold_df[f"label_{task}"].sum()
        gold_total = len(gold_df)

        log.info(f"   {task.capitalize():>5} - Silver: {silver_pos:,}/{silver_total:,} ({silver_pos/silver_total:.1%}) | "
                 f"Gold: {gold_pos:,}/{gold_total:,} ({gold_pos/gold_total:.1%})")

    # Main training loop
    task_progress = tqdm(["keto", "vegan"], desc="🔬 Training Tasks",
                         position=0, leave=True,
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for task in task_progress:
        task_start = time.time()
        task_progress.set_description(f"🔬 Training {task.capitalize()}")

        # Extract labels
        y_train = silver_df[f"silver_{task}"].values
        y_true = gold_df[f"label_{task}"].values

        log.info(f"\n🎯 Processing {task.upper()} classification:")
        log.info(f"   Training labels - Positive: {y_train.sum():,} ({y_train.mean():.1%})")
        log.info(f"   Test labels - Positive: {y_true.sum():,} ({y_true.mean():.1%})")

        # Handle class imbalance
        if apply_smote:
            smote_start = time.time()
            original_size = len(y_train)

            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                log.warning(f"   ⚠️  Only one class present in {task} training data, skipping SMOTE")
                X_train = X_silver
            else:
                minority_ratio = min(np.bincount(y_train)) / len(y_train)
                log.info(f"   Minority class ratio: {minority_ratio:.1%}")

                if minority_ratio < 0.4:
                    log.info(f"   🔄 Applying SMOTE (minority < 40%)...")
                    try:
                        with tqdm(total=1, desc="   ├─ SMOTE Processing",
                                  position=1, leave=False,
                                  bar_format="   ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as smote_pbar:

                            X_train, y_train = apply_smote(X_silver, y_train)
                            smote_pbar.update(1)

                        smote_time = time.time() - smote_start
                        new_size = len(y_train)
                        new_ratio = min(np.bincount(y_train)) / len(y_train)

                        log.info(f"   ✅ SMOTE completed in {smote_time:.1f}s")
                        log.info(f"   ├─ Size: {original_size:,} → {new_size:,} ({new_size/original_size:.1f}x)")
                        log.info(f"   └─ Minority ratio: {minority_ratio:.1%} → {new_ratio:.1%}")

                    except Exception as e:
                        log.warning(f"   ❌ SMOTE failed for {task}: {str(e)[:60]}...")
                        log.info(f"   └─ Falling back to original data")
                        X_train = X_silver
                else:
                    log.info(f"   ✅ Classes already balanced, skipping SMOTE")
                    X_train = X_silver
        else:
            log.info(f"   ⏭️  SMOTE disabled, using original data")
            X_train = X_silver

        # Build and train models
        models = build_models(task, domain)

        # Filter out Rule model for image domain
        if domain == "image":
            models = {k: v for k, v in models.items() if k != "Rule"}

        log.info(f"   🤖 Training {len(models)} models: {list(models.keys())}")

        best_f1, best_res = -1.0, None
        model_results = []

        # Model training progress
        model_progress = tqdm(models.items(), desc="   ├─ Training Models",
                              position=1, leave=False,
                              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]")

        for name, base in model_progress:
            model_start = time.time()
            model_progress.set_description(f"   ├─ Training {name}")

            try:
                # Check for single-class case
                if len(np.unique(y_train)) < 2:
                    log.warning(f"      ⚠️  {name}: Only one class in training data, skipping")
                    continue

                # Model training phases
                with tqdm(total=4, desc=f"      ├─ {name}", position=2, leave=False,
                          bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as model_pbar:

                    # Step 1: Model fitting
                    model_pbar.set_description(f"      ├─ {name}: Fitting")
                    model = clone(base)

                    # Memory efficiency check
                    if hasattr(X_train, "toarray") and X_train.shape[1] > 10000:
                        log.debug(f"         ├─ {name}: Processing large sparse matrix")

                    model.fit(X_train, y_train)
                    model_pbar.update(1)

                    # Step 2: Ensure probabilistic predictions
                    model_pbar.set_description(f"      ├─ {name}: Configuring")
                    model = ensure_predict_proba(model, X_train, y_train)
                    model_pbar.update(1)

                    # Step 3: Generate predictions
                    model_pbar.set_description(f"      ├─ {name}: Predicting")
                    try:
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(X_gold)[:, 1]
                        elif hasattr(model, "decision_function"):
                            scores = model.decision_function(X_gold)
                            prob = 1 / (1 + np.exp(-scores))  # Sigmoid
                        else:
                            # Fallback to binary predictions
                            pred_binary = model.predict(X_gold)
                            prob = pred_binary.astype(float)
                            log.warning(f"      ⚠️  {name}: Using binary predictions (suboptimal)")
                    except Exception as pred_error:
                        log.error(f"      ❌ {name}: Prediction failed - {str(pred_error)[:40]}...")
                        continue
                    model_pbar.update(1)

                    # Step 4: Apply rule verification
                    model_pbar.set_description(f"      ├─ {name}: Verifying")
                    prob = verify_with_rules(task, gold_clean, prob)
                    pred = (prob >= 0.5).astype(int)
                    model_pbar.update(1)

                # Calculate metrics
                model_time = time.time() - model_start
                model_name_with_domain = f"{name}_{domain.upper()}"

                res = dict(
                    task=task,
                    model=model_name_with_domain,
                    ACC=accuracy_score(y_true, pred),
                    PREC=precision_score(y_true, pred, zero_division=0),
                    REC=recall_score(y_true, pred, zero_division=0),
                    F1=f1_score(y_true, pred, zero_division=0),
                    ROC=roc_auc_score(y_true, prob),
                    PR=average_precision_score(y_true, prob),
                    prob=prob,
                    pred=pred,
                    training_time=model_time,
                    domain=domain
                )

                model_results.append(res)

                # Log performance
                log.info(f"      ✅ {model_name_with_domain:>12}: F1={res['F1']:.3f} | "
                         f"ACC={res['ACC']:.3f} | PREC={res['PREC']:.3f} | "
                         f"REC={res['REC']:.3f} | Time={model_time:.1f}s")

                # Track best model
                if res["F1"] > best_f1:
                    best_f1, best_res = res["F1"], res
                    BEST[name] = model  # Store without domain suffix
                    log.info(f"      🏆 New best model for {task}: {model_name_with_domain} (F1={best_f1:.3f})")

            except Exception as e:
                model_time = time.time() - model_start
                log.error(f"      ❌ {name:>8}: FAILED after {model_time:.1f}s - {str(e)[:50]}...")

                if log.level <= logging.DEBUG:
                    import traceback
                    log.debug(f"Full traceback for {name}:\n{traceback.format_exc()}")

        # Fallback handling
        if best_res is None:
            log.warning(f"   ⚠️  All models failed for {task}! Using RuleModel fallback...")

            fallback_start = time.time()
            rule = build_models(task, domain="text")["Rule"]

            with tqdm(total=1, desc="   ├─ Rule Fallback", position=1, leave=False,
                      bar_format="   ├─ {desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as rule_pbar:
                prob = rule.predict_proba(gold_clean)[:, 1]
                pred = (prob >= 0.5).astype(int)
                rule_pbar.update(1)

            fallback_time = time.time() - fallback_start
            best_res = pack(y_true, prob) | dict(
                task=task, model=f"Rule_{domain.upper()}", prob=prob, pred=pred,
                training_time=fallback_time, domain=domain
            )
            BEST[task] = rule

            log.info(f"   🛡️  Rule fallback: F1={best_res['F1']:.3f} | Time={fallback_time:.1f}s")

        # Task completion summary
        task_time = time.time() - task_start
        results.append(best_res)

        log.info(f"   🎯 {task.upper()} COMPLETE:")
        log.info(f"   ├─ Best Model: {best_res['model']} (F1={best_res['F1']:.3f})")
        log.info(f"   ├─ Final Metrics: ACC={best_res['ACC']:.3f} | "
                 f"PREC={best_res['PREC']:.3f} | REC={best_res['REC']:.3f}")
        log.info(f"   └─ Task Time: {task_time:.1f}s")

        # Update progress
        task_progress.set_postfix({
            'Best': best_res['model'],
            'F1': f"{best_res['F1']:.3f}",
            'Time': f"{task_time:.1f}s"
        })

    # Pipeline completion
    pipeline_time = time.time() - pipeline_start

    log.info(f"\n🏁 MODE A PIPELINE COMPLETE:")
    log.info(f"   ├─ Total Time: {pipeline_time:.1f}s")
    log.info(f"   ├─ Tasks Completed: {len(results)}")
    log.info(f"   └─ Domain: {domain}")

    # Summary table
    log.info(f"\n📊 FINAL RESULTS SUMMARY:")
    for i, res in enumerate(results, 1):
        log.info(f"   {i}. {res['task'].upper():>5} | {res['model']:>15} | "
                 f"F1={res['F1']:.3f} | ACC={res['ACC']:.3f} | "
                 f"Time={res.get('training_time', 0):.1f}s")

    # Display formatted table
    table("MODE A (silver → gold)", results)

    return results


# =============================================================================
# METRICS AND VISUALIZATION
# =============================================================================
"""
Functions for calculating metrics and displaying results.
"""

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
    print(f"\n╭─ {title} {'─' * (pad - len(title) - 2)}")
    print(hdr)
    print("├" + "─" * (len(hdr) - 2) + "┤")
    for r in rows:
        vals = " ".join(f"{r[c]:>7.2f}" for c in cols)
        print(f"│ {r['model']:<7} {r['task']:<5} {vals} │")
    print("╰" + "─" * (len(hdr) - 2) + "╯")


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
    Extract ResNet-50 embeddings for recipe images with comprehensive monitoring.
    
    This function implements a sophisticated image feature extraction pipeline:
    1. Checks for cached embeddings to avoid recomputation
    2. Loads pre-trained ResNet-50 (without classification head)
    3. Processes images in batches for efficiency
    4. Handles errors gracefully with detailed logging
    5. Applies quality filtering to remove bad images
    6. Saves embeddings with backup for reliability
    
    Args:
        df: DataFrame with image indices
        mode: Mode identifier ('silver', 'gold', etc.)
        force: Force recomputation even if cache exists
        
    Returns:
        Tuple of (embeddings_array, valid_indices)
        - embeddings_array: NumPy array of shape (n_images, 2048)
        - valid_indices: List of indices that have valid embeddings
        
    Note:
        The function returns both embeddings and indices to maintain
        alignment with the original dataset after filtering.
    """
    import time
    import os
    from collections import defaultdict, Counter
    from PIL import ImageStat
    import gc

    embedding_start = time.time()

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
        log.info(f"   └─ Zero vector shape: ({len(df)}, 2048)")
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
    if device_info['cuda_available']:
        log.info(f"   ├─ GPU count: {device_info['device_count']}")
        log.info(f"   └─ GPU name: {device_info['device_name']}")
    else:
        log.info(f"   └─ Using CPU (warning: much slower)")

    # Set up paths
    img_dir = CFG.image_dir / mode
    embed_path = img_dir / "embeddings.npy"
    backup_path = Path(f"embeddings_{mode}_backup.npy")
    metadata_path = img_dir / "embedding_metadata.json"

    log.info(f"   📁 Paths:")
    log.info(f"   ├─ Image directory: {img_dir}")
    log.info(f"   ├─ Cache file: {embed_path}")
    log.info(f"   └─ Backup file: {backup_path}")

    # ------------------------------------------------------------------
    # Cache Loading and Validation
    # ------------------------------------------------------------------
    if not force:
        log.info(f"\n   🔍 Cache Validation:")

        cache_options = [
            ("Primary cache", embed_path),
            ("Backup cache", backup_path)
        ]

        with tqdm(cache_options, desc="      ├─ Checking caches", position=1, leave=False,
                  bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as cache_pbar:

            for cache_name, cache_path in cache_pbar:
                cache_pbar.set_description(f"      ├─ Checking {cache_name.lower()}")

                if cache_path.exists():
                    try:
                        cache_start = time.time()
                        emb = np.load(cache_path)
                        load_time = time.time() - cache_start

                        log.info(f"      ├─ {cache_name}: Found ({emb.shape}) - loaded in {load_time:.2f}s")

                        if emb.shape[0] == len(df):
                            log.info(f"      ✅ {cache_name} matches target size - using cached embeddings")

                            # Load metadata if available
                            if metadata_path.exists():
                                try:
                                    with open(metadata_path, 'r') as f:
                                        metadata = json.load(f)
                                    log.info(f"      ├─ Cache metadata: {metadata.get('creation_time', 'Unknown time')}")
                                    log.info(f"      └─ Original stats: {metadata.get('success', '?')} success, "
                                             f"{metadata.get('failed', '?')} failed")
                                except Exception as e:
                                    log.debug(f"      └─ Metadata load failed: {e}")

                            return emb, list(df.index)

                        else:
                            log.warning(f"      ⚠️  {cache_name} size mismatch: {emb.shape[0]} != {len(df)}")

                            # Truncate oversize cache if possible
                            if emb.shape[0] > len(df):
                                log.info(f"      ├─ Truncating cache from {emb.shape[0]} to {len(df)}")
                                return emb[:len(df)], list(df.index)

                    except Exception as e:
                        log.error(f"      ❌ {cache_name} load failed: {str(e)[:60]}...")
                else:
                    log.info(f"      ├─ {cache_name}: Not found")

        log.info(f"      └─ No valid cache found - will compute embeddings")

    else:
        log.info(f"\n   🔄 Cache bypassed (force=True) - recomputing embeddings")

    # ------------------------------------------------------------------
    # Pre-processing Analysis
    # ------------------------------------------------------------------
    log.info(f"\n   📊 Pre-processing Analysis:")

    with tqdm(total=3, desc="      ├─ Analyzing images", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as analysis_pbar:

        analysis_pbar.set_description("      ├─ Scanning directory")

        # Check which images exist
        existing_images = []
        missing_images = []
        corrupted_images = []

        for idx in df.index:
            img_file = img_dir / f"{idx}.jpg"
            if img_file.exists():
                try:
                    # Quick validation
                    with Image.open(img_file) as img:
                        img.verify()  # Verify image integrity
                    existing_images.append(idx)
                except Exception:
                    corrupted_images.append(idx)
            else:
                missing_images.append(idx)

        analysis_pbar.update(1)

        analysis_pbar.set_description("      ├─ Computing statistics")

        # Calculate statistics
        total_images = len(df)
        existing_count = len(existing_images)
        missing_count = len(missing_images)
        corrupted_count = len(corrupted_images)

        log.info(f"      📈 Image Availability:")
        log.info(f"      ├─ Total expected: {total_images:,}")
        log.info(f"      ├─ Available: {existing_count:,} ({existing_count/total_images*100:.1f}%)")
        log.info(f"      ├─ Missing: {missing_count:,} ({missing_count/total_images*100:.1f}%)")
        log.info(f"      └─ Corrupted: {corrupted_count:,} ({corrupted_count/total_images*100:.1f}%)")

        analysis_pbar.update(1)

        # Sample image analysis
        analysis_pbar.set_description("      ├─ Sampling quality")

        if existing_images:
            sample_size = min(100, len(existing_images))
            sample_indices = np.random.choice(existing_images, sample_size, replace=False)

            image_stats = {
                'sizes': [],
                'modes': Counter(),
                'formats': Counter(),
                'file_sizes': []
            }

            # Analyze sample for statistics
            for idx in sample_indices[:10]:
                img_file = img_dir / f"{idx}.jpg"
                try:
                    with Image.open(img_file) as img:
                        image_stats['sizes'].append(img.size)
                        image_stats['modes'][img.mode] += 1
                        image_stats['formats'][img.format] += 1
                        image_stats['file_sizes'].append(img_file.stat().st_size)
                except Exception:
                    pass

            if image_stats['sizes']:
                avg_width = sum(s[0] for s in image_stats['sizes']) / len(image_stats['sizes'])
                avg_height = sum(s[1] for s in image_stats['sizes']) / len(image_stats['sizes'])
                avg_file_size = sum(image_stats['file_sizes']) / len(image_stats['file_sizes'])

                log.info(f"      📊 Sample Analysis ({len(image_stats['sizes'])} images):")
                log.info(f"      ├─ Average size: {avg_width:.0f}×{avg_height:.0f} pixels")
                log.info(f"      ├─ Average file size: {avg_file_size/1024:.1f} KB")
                log.info(f"      ├─ Color modes: {dict(image_stats['modes'])}")
                log.info(f"      └─ Formats: {dict(image_stats['formats'])}")

        analysis_pbar.update(1)

    # ------------------------------------------------------------------
    # Model Setup and Initialization
    # ------------------------------------------------------------------
    log.info(f"\n   🤖 Model Setup:")

    with tqdm(total=4, desc="      ├─ Loading model", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as model_pbar:

        model_pbar.set_description("      ├─ Loading ResNet-50")
        model_start = time.time()

        # Load pre-trained ResNet-50
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model_pbar.update(1)

        model_pbar.set_description("      ├─ Modifying architecture")
        # Remove classification head for feature extraction
        model.fc = torch.nn.Identity()
        model.eval()  # Set to evaluation mode
        model_pbar.update(1)

        model_pbar.set_description("      ├─ Moving to device")
        model.to(device_info['device'])
        model_time = time.time() - model_start
        model_pbar.update(1)

        model_pbar.set_description("      ├─ Setting up preprocessing")
        # Standard ImageNet preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        model_pbar.update(1)

    log.info(f"      ✅ Model loaded in {model_time:.2f}s")
    log.info(f"      ├─ Architecture: ResNet-50 (feature extractor)")
    log.info(f"      ├─ Output dimension: 2048")
    log.info(f"      ├─ Device: {device_info['device']}")
    log.info(f"      └─ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Memory usage after model loading
    if device_info['cuda_available']:
        gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        log.info(f"      📊 GPU memory allocated: {gpu_memory:.1f} MB")

    # ------------------------------------------------------------------
    # Embedding Extraction with Detailed Progress
    # ------------------------------------------------------------------
    log.info(f"\n   ⚡ Feature Extraction:")

    vectors = []
    processing_stats = {
        'success': 0,
        'missing': 0,
        'failed': 0,
        'processing_times': [],
        'error_types': Counter(),
        'batch_times': []
    }

    failed_details = []

    # Determine batch size based on available memory
    if device_info['cuda_available']:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = max(1, min(32, int(gpu_memory_gb * 2)))  # Conservative estimate
    else:
        batch_size = 8  # Conservative CPU batch size

    log.info(f"      🔧 Processing Configuration:")
    log.info(f"      ├─ Batch size: {batch_size}")
    log.info(f"      ├─ Total batches: {(len(df) + batch_size - 1) // batch_size}")
    log.info(f"      └─ Expected output shape: ({len(df)}, 2048)")

    # Main processing loop
    with tqdm(df.index, desc=f"      ├─ Extracting {mode} embeddings",
              position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}] {rate_fmt}") as extract_pbar:

        for i, idx in enumerate(extract_pbar):
            process_start = time.time()
            img_file = img_dir / f"{idx}.jpg"

            # Update progress bar periodically
            if i % 100 == 0:
                success_rate = processing_stats['success'] / max(1, i) * 100
                extract_pbar.set_postfix({
                    'Success': f"{processing_stats['success']}",
                    'Failed': f"{processing_stats['failed']}",
                    'Missing': f"{processing_stats['missing']}",
                    'Rate': f"{success_rate:.1f}%"
                })

            # Check if image exists
            if not img_file.exists():
                processing_stats['missing'] += 1
                vectors.append(np.zeros(2048, dtype=np.float32))
                continue

            try:
                # Load and preprocess image
                img = Image.open(img_file).convert('RGB')

                # Log properties for first few images
                if processing_stats['success'] < 5:
                    log.debug(f"         ├─ Processing {img_file.name}: {img.size} {img.mode}")

                with torch.no_grad():
                    # Preprocess and add batch dimension
                    tensor = preprocess(img).unsqueeze(0).to(device_info['device'])

                    # Extract features
                    features = model(tensor).squeeze().cpu().numpy()

                    # Validate output shape
                    if features.shape != (2048,):
                        raise ValueError(f"Unexpected feature shape: {features.shape}")

                vectors.append(features)
                processing_stats['success'] += 1

                # Track processing time
                process_time = time.time() - process_start
                processing_stats['processing_times'].append(process_time)

            except Exception as e:
                processing_stats['failed'] += 1

                # Categorize error
                error_type = type(e).__name__
                processing_stats['error_types'][error_type] += 1

                # Log error details
                error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                failed_details.append((idx, img_file, error_type, error_msg))

                # Log first few errors
                if processing_stats['failed'] <= 5:
                    log.warning(f"         ❌ Failed {img_file.name}: {error_type} - {error_msg}")

                # Add zero vector for failed processing
                vectors.append(np.zeros(2048, dtype=np.float32))

            # Periodic memory cleanup
            if i % 1000 == 0 and i > 0:
                gc.collect()
                optimize_memory_usage("Batch Processing")
                if device_info['cuda_available']:
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Post-processing and Results Analysis
    # ------------------------------------------------------------------
    extraction_time = time.time() - embedding_start

    log.info(f"\n   📊 Extraction Results:")
    log.info(f"   ├─ Total processing time: {extraction_time:.1f}s")
    log.info(f"   ├─ Successfully processed: {processing_stats['success']:,}")
    log.info(f"   ├─ Missing images: {processing_stats['missing']:,}")
    log.info(f"   ├─ Failed processing: {processing_stats['failed']:,}")
    log.info(f"   └─ Overall success rate: {processing_stats['success']/len(df)*100:.1f}%")

    # Performance metrics
    if processing_stats['processing_times']:
        avg_time = sum(processing_stats['processing_times']) / len(processing_stats['processing_times'])
        throughput = processing_stats['success'] / extraction_time

        log.info(f"   ⚡ Performance Metrics:")
        log.info(f"   ├─ Average processing time: {avg_time:.3f}s per image")
        log.info(f"   ├─ Throughput: {throughput:.1f} images/second")
        log.info(f"   └─ Device efficiency: {device_info['device']}")

    # Error analysis
    if processing_stats['failed'] > 0:
        log.info(f"   ⚠️  Error Analysis:")
        total_errors = sum(processing_stats['error_types'].values())

        for error_type, count in processing_stats['error_types'].most_common():
            percentage = count / total_errors * 100
            log.info(f"   ├─ {error_type}: {count} ({percentage:.1f}%)")

        # Save error log
        if failed_details:
            error_log_path = img_dir / "embedding_errors.txt"
            try:
                with open(error_log_path, "w") as f:
                    f.write("Index\tFile\tErrorType\tErrorMessage\n")
                    for idx, img_file, error_type, error_msg in failed_details:
                        f.write(f"{idx}\t{img_file.name}\t{error_type}\t{error_msg}\n")
                log.info(f"   💾 Error details saved to: {error_log_path}")
            except Exception as e:
                log.warning(f"   ⚠️  Failed to save error log: {e}")

    # ------------------------------------------------------------------
    # Save Results and Metadata
    # ------------------------------------------------------------------
    log.info(f"\n   💾 Saving Results:")

    with tqdm(total=4, desc="      ├─ Saving files", position=1, leave=False,
              bar_format="      ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as save_pbar:

        save_pbar.set_description("      ├─ Stacking vectors")
        # Convert list to numpy array
        arr = np.vstack(vectors)
        save_pbar.update(1)

        save_pbar.set_description("      ├─ Creating directories")
        # Ensure directory exists
        embed_path.parent.mkdir(parents=True, exist_ok=True)
        save_pbar.update(1)

        save_pbar.set_description("      ├─ Saving primary cache")
        # Save primary cache
        np.save(embed_path, arr)
        save_pbar.update(1)

        save_pbar.set_description("      ├─ Saving backup")
        # Save backup
        np.save(backup_path, arr)
        save_pbar.update(1)

    # Save metadata
    metadata = {
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'mode': mode,
        'total_images': len(df),
        'success': processing_stats['success'],
        'missing': processing_stats['missing'],
        'failed': processing_stats['failed'],
        'processing_time_seconds': extraction_time,
        'device': str(device_info['device']),
        'model': 'ResNet-50',
        'output_shape': list(arr.shape),
        'avg_processing_time': avg_time if 'avg_time' in locals() else 0,
        'throughput_images_per_second': throughput if 'throughput' in locals() else 0
    }

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        log.info(f"   💾 Metadata saved to: {metadata_path}")
    except Exception as e:
        log.warning(f"   ⚠️  Failed to save metadata: {e}")

    # File size information
    try:
        primary_size = embed_path.stat().st_size / (1024**2)
        backup_size = backup_path.stat().st_size / (1024**2)

        log.info(f"   📊 File Information:")
        log.info(f"   ├─ Primary cache: {primary_size:.1f} MB ({embed_path})")
        log.info(f"   ├─ Backup cache: {backup_size:.1f} MB ({backup_path})")
        log.info(f"   └─ Array shape: {arr.shape}")

    except Exception as e:
        log.warning(f"   ⚠️  File size analysis failed: {e}")

    # Memory cleanup
    del model, vectors
    gc.collect()
    if device_info['cuda_available']:
        torch.cuda.empty_cache()
        final_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
        log.info(f"   🧹 GPU memory after cleanup: {final_gpu_memory:.1f} MB")

    # ------------------------------------------------------------------
    # Final Summary and Quality Filtering
    # ------------------------------------------------------------------
    log.info(f"\n   🏁 EMBEDDING EXTRACTION COMPLETE:")
    log.info(f"   ├─ Output shape: {arr.shape}")
    log.info(f"   ├─ Success rate: {processing_stats['success']/len(df)*100:.1f}%")
    log.info(f"   ├─ Total time: {extraction_time:.1f}s")
    log.info(f"   ├─ Throughput: {processing_stats['success']/extraction_time:.1f} images/s")
    log.info(f"   └─ Files saved: Primary + Backup + Metadata")

    # Apply quality filtering
    original_indices = list(df.index)
    if arr.shape[0] > 10:  # Only filter if we have enough images
        arr, valid_indices = filter_low_quality_images(img_dir, arr, original_indices)
        if len(valid_indices) != len(original_indices):
            log.info(f"   📊 Quality filtering reduced images from {len(original_indices)} to {len(valid_indices)}")
    else:
        valid_indices = original_indices

    return arr, valid_indices


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================
"""
Advanced ensemble techniques for combining predictions from multiple models.
These methods improve performance by leveraging the strengths of different
models and feature types.
"""

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
    
    This simple ensemble method combines the best performing model
    from each domain using a weighted average.
    
    Args:
        task: 'keto' or 'vegan'
        text_results: Results from text-based models
        image_results: Results from image-based models
        gold_df: Gold standard DataFrame
        alpha: Weight for image predictions (0=text only, 1=image only)
        
    Returns:
        Dictionary with ensemble results
    """
    # Find best models from each domain
    best_text = max(
        (r for r in text_results if r["task"] == task),
        key=lambda r: r["F1"]
    )
    best_img = max(
        (r for r in image_results if r["task"] == task and len(r.get("prob", [])) > 0),
        key=lambda r: r["F1"],
        default=None
    )

    log.info(
        f"▶️  BEST-TWO ({task}) — text={best_text['model']} "
        f"(F1={best_text['F1']:.3f}), "
        f"image={best_img['model'] if best_img else 'N/A'}"
    )

    # Align probability vectors
    txt_prob = pd.Series(best_text["prob"], index=gold_df.index)

    if best_img is None:  # No image model available
        final_prob = txt_prob.values
        rows_with_img = []
    else:
        img_idx = gold_df.index[:len(best_img["prob"])]
        img_prob = pd.Series(best_img["prob"], index=img_idx)

        rows_with_img = img_idx
        final_prob = txt_prob.copy()
        # Weighted average for rows with images
        final_prob.loc[rows_with_img] = (
            alpha * img_prob.loc[rows_with_img]
            + (1 - alpha) * txt_prob.loc[rows_with_img]
        )
        final_prob = final_prob.values

    # Apply verification and calculate metrics
    final_prob = verify_with_rules(task, gold_df.clean, final_prob)
    final_pred = (final_prob >= 0.5).astype(int)
    y_true = gold_df[f"label_{task}"].values

    return pack(y_true, final_prob) | {
        "model": f"BestTwo(alpha={alpha})",
        "task": task,
        "prob": final_prob,
        "pred": final_pred,
        "text_model": best_text["model"],
        "image_model": best_img["model"] if best_img else None,
        "alpha": alpha,
        "rows_image": len(rows_with_img),
        "rows_text_only": len(gold_df) - len(rows_with_img),
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
    text_only_mask = gold.get("has_image", np.ones(len(X_gold), dtype=bool)) == False

    # Compute dynamic weights
    weights = []
    for i, (name, _) in enumerate(estimators):
        is_image_model = "IMAGE" in name.upper()
        row_weights = np.ones(len(X_gold))
        if is_image_model:
            row_weights[text_only_mask] = 0  # Suppress image models for text-only rows
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
    import time
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
            log.info(f"   ✅ Loaded saved parameters for {len(saved_params)} models")
            for model_name in saved_params:
                log.info(f"      ├─ {model_name}: {saved_params[model_name]}")
        except FileNotFoundError:
            log.warning(f"   ⚠️  best_params.json not found, using default hyperparameters")
        except Exception as e:
            log.error(f"   ❌ Error loading parameters: {e}")

    # Filter and rank available models
    available_models = [r for r in res if r["task"] == task and r["model"] != "Rule"]

    if not available_models:
        log.error(f"   ❌ No models available for {task} ensemble")
        raise ValueError(f"No models available for {task}")

    if len(available_models) < n:
        log.warning(f"   ⚠️  Only {len(available_models)} models available, requested {n}")
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
                    log.info(f"         ├─ Applied saved parameters: {saved_params[base_key]}")
                else:
                    log.info(f"         ├─ Tuning hyperparameters...")
                    base = tune(base_key, base, X_vec, silver[f"silver_{task}"])
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

                log.info(f"         ├─ Individual performance: F1={individual_f1:.3f}, ACC={individual_acc:.3f}")
                model_pbar.update(1)

                # Log false predictions
                model_pbar.set_description(f"         ├─ {name}: Analyzing")
                log_false_preds(task, gold.clean, y_true, y_pred_i, model_name=name)

                # Ensure probability predictions
                if not hasattr(base, "predict_proba"):
                    log.info(f"         ├─ Adding probability calibration to {name}")
                    base = CalibratedClassifierCV(base, cv=3, method='sigmoid')
                    base.fit(X_vec, silver[f"silver_{task}"])

                model_pbar.update(1)

            # Record successful preparation
            model_time = time.time() - model_start
            preparation_times[name] = model_time
            estimators.append((name, base))

            log.info(f"      ✅ {name} prepared successfully in {model_time:.1f}s")

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

            log.error(f"      ❌ {name} failed after {model_time:.1f}s: {error_msg}")

            if log.level <= logging.DEBUG:
                import traceback
                log.debug(f"Full traceback for {name}:\n{traceback.format_exc()}")

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
        raise RuntimeError(f"No models successfully prepared for {task} ensemble")

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
        log.info(f"   ✅ Soft voting ensemble created in {ensemble_create_time:.1f}s")
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
                        log.debug(f"      ├─ {name}: decision_function + sigmoid")
                    else:
                        binary_preds = clf.predict(X_gold).astype(float)
                        probs.append(binary_preds)
                        log.warning(f"      ├─ {name}: using binary predictions (suboptimal)")

                except Exception as pred_error:
                    averaging_errors.append((name, str(pred_error)[:40]))
                    log.error(f"      ├─ {name}: prediction failed - {str(pred_error)[:40]}...")

        if not probs:
            raise RuntimeError("All models failed to generate predictions for ensemble")

        # Average probabilities
        prob = np.mean(probs, axis=0)

        prob_time = time.time() - prob_start
        log.info(f"   ✅ Manual averaging completed in {prob_time:.1f}s")
        log.info(f"      ├─ Successfully averaged: {len(probs)}/{len(estimators)} models")

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

        log.info(f"   ├─ Best individual: {best_individual[0]['model']} (F1={best_individual_f1:.3f})")
        log.info(f"   ├─ Ensemble F1: {ensemble_metrics['f1']:.3f}")
        log.info(f"   └─ Improvement: {ensemble_improvement:+.3f} ({ensemble_improvement/best_individual_f1*100:+.1f}%)")

    # Error analysis
    log.info(f"\n   🔍 Error Analysis:")
    log_false_preds(task, gold.clean, y_true, y_pred, model_name=f"EnsembleTop{actual_n}")

    # Confidence analysis
    confidence_high = (np.abs(prob - 0.5) > 0.3).sum()
    confidence_medium = (np.abs(prob - 0.5) > 0.1).sum() - confidence_high
    confidence_low = len(prob) - confidence_high - confidence_medium

    log.info(f"   ├─ High confidence (>0.8 or <0.2): {confidence_high} ({confidence_high/len(prob)*100:.1f}%)")
    log.info(f"   ├─ Medium confidence (0.6-0.8, 0.2-0.4): {confidence_medium} ({confidence_medium/len(prob)*100:.1f}%)")
    log.info(f"   └─ Low confidence (0.4-0.6): {confidence_low} ({confidence_low/len(prob)*100:.1f}%)")

    # Final summary
    total_time = time.time() - ensemble_start

    log.info(f"\n   🏁 ENSEMBLE COMPLETE:")
    log.info(f"   ├─ Ensemble method: {ensemble_method}")
    log.info(f"   ├─ Models used: {actual_n}/{n}")
    log.info(f"   ├─ Final F1 score: {ensemble_metrics['f1']:.3f}")
    log.info(f"   ├─ Rule changes: {verification_changes}")
    log.info(f"   └─ Total time: {total_time:.1f}s")

    # Return results
    return pack(y_true, prob) | {
        "model": f"Ens{actual_n}",
        "task": task,
        "prob": prob,
        "pred": y_pred,
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
    import time
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
            log.info(f"   ⏭️  No image models for {task}; using text ensemble only")
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
        for α in alphas:
            blend = txt_prob.copy()
            blend.loc[rows_img] = α * img_prob.loc[rows_img] + (1-α) * txt_prob.loc[rows_img]
            f1 = f1_score(y_true, (blend.values >= .5).astype(int), zero_division=0)
            if f1 > best_f1:
                best_alpha, best_f1, best_prob = α, f1, blend.values

        # Apply verification and return
        best_prob = verify_with_rules(task, gold.clean, best_prob)
        best_pred = (best_prob >= .5).astype(int)

        return pack(y_true, best_prob) | {
            "model": f"SmartEns(Text={text_best['model']},Img={image_best['model']},α={best_alpha})",
            "task": task,
            "prob": best_prob,
            "pred": best_pred,
            "alpha": best_alpha,
            "rows_image": len(rows_img),
            "rows_text_only": len(rows_txt),
            "text_model": text_best["model"],
            "image_model": image_best["model"],
        }

    # Extract available models
    model_names = [r["model"] for r in res if r["task"] == task and r["model"] != "Rule"]
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
        single_model = [r for r in res if r["task"] == task and r["model"] != "Rule"][0]
        log.info(f"   ⚠️  Only one model available: {single_model['model']}")
        log.info(f"   └─ F1={single_model['F1']:.3f}, skipping ensemble optimization")
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
        log.warning(f"   ⚠️  Weights sum to {weight_sum:.3f}, normalizing to 1.0")
        weights = {k: v/weight_sum for k, v in weights.items()}

    log.info(f"   ├─ Weights: {', '.join([f'{k}={v:.3f}' for k, v in weights.items()])}")

    # Show individual model performance
    individual_models = [r for r in res if r["task"] == task and r["model"] != "Rule"]

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
                result = top_n(task, res, X_vec, clean, X_gold, silver, gold, n=n)
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
                log.info(f"      🏆 New best ensemble: n={n} (Composite={best_score:.3f})")

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
                log.debug(f"Full traceback for ensemble n={n}:\n{traceback.format_exc()}")

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
            log.info(f"   ├─ Best Individual: {best_individual['model']} (F1={best_individual['F1']:.3f})")
            log.info(f"   ├─ Best Ensemble: {best_result['model']} (F1={best_result['F1']:.3f})")
            log.info(f"   └─ F1 Improvement: {f1_improvement:+.3f} ({f1_improvement/best_individual['F1']*100:+.1f}%)")

        # Ensemble composition
        if 'Ens' in best_result['model']:
            ensemble_size = int(best_result['model'][-1])
            log.info(f"\n   🔧 Ensemble Composition (Top {ensemble_size}):")
            top_models = sorted(individual_models, key=lambda x: x['F1'], reverse=True)[:ensemble_size]
            for i, model_res in enumerate(top_models, 1):
                log.info(f"   ├─ {i}. {model_res['model']:>8} (F1={model_res['F1']:.3f})")

    else:
        log.error(f"\n   ❌ ENSEMBLE OPTIMIZATION FAILED:")
        log.error(f"   ├─ No valid ensembles found")
        log.error(f"   ├─ Available models: {len(unique_models)}")
        log.error(f"   └─ Total Time: {total_time:.1f}s")

        # Fallback
        if individual_models:
            best_individual = max(individual_models, key=lambda x: x['F1'])
            log.info(f"   🛡️  Falling back to best individual model:")
            log.info(f"   └─ {best_individual['model']} (F1={best_individual['F1']:.3f})")
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

    return best_result

# =============================================================================
# EVALUATION EXPORTS
# =============================================================================
"""
Functions for exporting evaluation results and visualizations.
"""

def export_eval_plots(results: list[dict], gold_df: pd.DataFrame,
                      out_dir: Path = Path("plots")) -> None:
    """
    Export evaluation plots and metrics for all models.
    
    Creates confusion matrices and ROC curves for each model,
    along with a CSV summary of all metrics.
    
    Args:
        results: List of result dictionaries
        gold_df: Gold standard DataFrame
        out_dir: Directory for saving plots
    """
    out_dir.mkdir(exist_ok=True)
    rows = []

    for r in tqdm(results, desc="Saving plots and metrics"):
        task = r["task"]
        model = r["model"]
        prob = r.get("prob")
        pred = r.get("pred")

        row = dict(model=model, task=task,
                   accuracy=None, precision=None,
                   recall=None, F1=None, AUC=None)

        try:
            # Get true labels
            true_labels = gold_df[f"label_{task}"].values

            if pred is not None and len(pred) > 0:
                # Handle dimension mismatches
                if len(pred) != len(true_labels):
                    log.warning(f"   ⚠️  Dimension mismatch for {model}-{task}: "
                                f"pred={len(pred)}, true={len(true_labels)}")

                    min_len = min(len(pred), len(true_labels))
                    pred = pred[:min_len]
                    true_labels = true_labels[:min_len]
                    if prob is not None:
                        prob = prob[:min_len]

                    log.info(f"      ├─ Truncated to {min_len} samples")

                # Calculate metrics
                try:
                    row["accuracy"] = accuracy_score(true_labels, pred)
                    row["precision"] = precision_score(true_labels, pred, zero_division=0)
                    row["recall"] = recall_score(true_labels, pred, zero_division=0)
                    row["F1"] = f1_score(true_labels, pred, zero_division=0)

                    # Create confusion matrix plot
                    cm = confusion_matrix(true_labels, pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ConfusionMatrixDisplay(cm).plot(ax=ax)
                    ax.set_title(f"{model} – {task} – Confusion matrix")
                    plt.tight_layout()
                    plt.savefig(out_dir / f"{model}_{task}_cm.png", dpi=100, bbox_inches='tight')
                    plt.close(fig)

                except Exception as e:
                    log.warning(f"   ⚠️  Metrics calculation failed for {model}-{task}: {e}")

            if prob is not None and len(prob) > 0:
                try:
                    # Handle dimension mismatch
                    if len(prob) != len(true_labels):
                        min_len = min(len(prob), len(true_labels))
                        prob = prob[:min_len]
                        true_labels = true_labels[:min_len]

                    auc = roc_auc_score(true_labels, prob)
                    row["AUC"] = auc

                    # Create ROC curve plot
                    fpr, tpr, _ = roc_curve(true_labels, prob)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f"{model} – {task} – ROC Curve")
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(out_dir / f"{model}_{task}_roc.png", dpi=100, bbox_inches='tight')
                    plt.close(fig)

                except Exception as e:
                    log.warning(f"   ⚠️  ROC plot failed for {model}-{task}: {e}")

        except Exception as e:
            log.error(f"   ❌ Complete failure for {model}-{task}: {e}")

        rows.append(row)

    # Save results
    try:
        pd.DataFrame(rows).to_csv("evaluation_results.csv", index=False)
        log.info("   ✅ Saved evaluation_results.csv and plots")
    except Exception as e:
        log.error(f"   ❌ Failed to save results: {e}")


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
    import time
    import gc
    from datetime import datetime

    # Initialize pipeline tracking
    pipeline_start = time.time()

    # Log pipeline initialization
    log.info("🚀 STARTING FULL ML PIPELINE")
    log.info(f"   Mode: {mode}")
    log.info(f"   Force recomputation: {force}")
    log.info(f"   Sample fraction: {sample_frac or 'Full dataset'}")
    log.info(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"   Available CPU cores: {psutil.cpu_count()}")
    log.info(f"   Available memory: {psutil.virtual_memory().total // (1024**3)} GB")

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
    if sample_frac:
        original_txt_size = len(silver_txt)
        original_img_size = len(silver_img)

        # Sample both datasets consistently
        silver_txt = silver_txt.sample(frac=sample_frac, random_state=42).copy()

        if not silver_img.empty:
            # Get consistent sampling across modalities
            sampled_indices = silver_txt.index
            available_img_indices = silver_img.index
            common_indices = sampled_indices.intersection(available_img_indices)

            if len(common_indices) > 0:
                silver_img = silver_img.loc[common_indices].copy()
                log.info(f"   📉 Consistent sampling: Using {len(common_indices):,} common indices")
            else:
                silver_img = silver_img.sample(frac=sample_frac, random_state=42).copy()
                log.info(f"   📉 Separate sampling: No common indices found")

        sampled_txt_size = len(silver_txt)
        sampled_img_size = len(silver_img)

        log.info(f"   📉 Applied sampling before processing:")
        log.info(f"   ├─ Text: {original_txt_size:,} → {sampled_txt_size:,} rows ({sample_frac:.1%})")
        log.info(f"   └─ Images: {original_img_size:,} → {sampled_img_size:,} rows ({sample_frac:.1%})")

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
        log.warning(f"High memory usage after data loading: {psutil.virtual_memory().percent:.1f}%")

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
    log.info(f"   ├─ Sparsity: {(1 - X_text_silver.nnz / X_text_silver.size):.1%}")
    log.info(f"   └─ Memory usage: ~{X_text_silver.data.nbytes // (1024**2)} MB")

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

            # Download images
            img_pbar.set_description("   ├─ Downloading silver images")
            if not silver_img.empty:
                silver_downloaded = _download_images(silver_img, CFG.image_dir / "silver")
                log.info(f"      ├─ Silver download: {len(silver_downloaded):,}/{len(silver_img):,} successful")
            else:
                silver_downloaded = []
                log.info(f"      ├─ Silver download: No images to download")
            img_pbar.update(1)

            img_pbar.set_description("   ├─ Downloading gold images")
            if not gold_img.empty:
                gold_downloaded = _download_images(gold_img, CFG.image_dir / "gold")
                log.info(f"      ├─ Gold download: {len(gold_downloaded):,}/{len(gold_img):,} successful")
            else:
                gold_downloaded = []
                log.info(f"      ├─ Gold download: No images to download")
            img_pbar.update(1)

            img_pbar.set_description("   ├─ Filtering by downloads")
            if silver_downloaded:
                img_silver_df = filter_silver_by_downloaded_images(silver_img, CFG.image_dir)
                log.info(f"      ├─ Silver filtered: {len(img_silver_df):,} with valid images")
            else:
                img_silver_df = pd.DataFrame()
                log.info(f"      ├─ Silver filtered: Empty (no downloads)")

            img_gold_df = filter_photo_rows(gold_img) if gold_downloaded else pd.DataFrame()
            log.info(f"      ├─ Gold filtered: {len(img_gold_df):,} with valid images")
            img_pbar.update(1)

            # Extract embeddings with proper alignment
            img_pbar.set_description("   ├─ Building silver embeddings")
            if not img_silver_df.empty:
                img_silver, silver_valid_indices = build_image_embeddings(img_silver_df, "silver", force)

                # Filter DataFrame to match embeddings
                if len(silver_valid_indices) != len(img_silver_df):
                    img_silver_df = img_silver_df.loc[silver_valid_indices].copy()
                    log.info(f"      ├─ Silver DF filtered: {len(img_silver_df):,} rows match embeddings")

                log.info(f"      ├─ Silver embeddings: {img_silver.shape}")
                log.info(f"      ├─ Silver DataFrame: {len(img_silver_df):,} rows")
            else:
                img_silver = np.array([]).reshape(0, 2048)
                silver_valid_indices = []
                log.info(f"      ├─ Silver embeddings: Empty array (no valid images)")
            img_pbar.update(1)

            img_pbar.set_description("   ├─ Building gold embeddings")
            if not img_gold_df.empty:
                img_gold, gold_valid_indices = build_image_embeddings(img_gold_df, "gold", force)

                # Filter DataFrame to match embeddings
                if len(gold_valid_indices) != len(img_gold_df):
                    img_gold_df = img_gold_df.loc[gold_valid_indices].copy()
                    log.info(f"      ├─ Gold DF filtered: {len(img_gold_df):,} rows match embeddings")

                log.info(f"      ├─ Gold embeddings: {img_gold.shape}")
                log.info(f"      ├─ Gold DataFrame: {len(img_gold_df):,} rows")
            else:
                img_gold = np.array([]).reshape(0, 2048)
                gold_valid_indices = []
                log.info(f"      ├─ Gold embeddings: Empty array (no valid images)")
            img_pbar.update(1)

            img_pbar.set_description("   ├─ Saving embeddings")
            if img_gold.size > 0:
                joblib.dump(img_gold, "embeddings/img_gold.pkl")
                log.info(f"      ├─ Saved gold embeddings to embeddings/img_gold.pkl")
            else:
                log.info(f"      ├─ Skipped saving empty gold embeddings")
            img_pbar.update(1)

        # Verify dimensions
        if img_silver.size > 0:
            log.info(f"   🔍 DIMENSION VERIFICATION:")
            log.info(f"   ├─ Silver embeddings: {img_silver.shape}")
            log.info(f"   ├─ Silver DataFrame: {len(img_silver_df):,} rows")
            log.info(f"   ├─ Gold embeddings: {img_gold.shape}")
            log.info(f"   └─ Gold DataFrame: {len(img_gold_df):,} rows")

            # Ensure alignment
            assert img_silver.shape[0] == len(img_silver_df), f"Silver dimension mismatch"
            assert img_gold.shape[0] == len(img_gold_df), f"Gold dimension mismatch"
            log.info(f"   ✅ All dimensions verified!")

        # Convert to sparse for efficiency
        if img_silver.size > 0:
            X_img_silver = csr_matrix(img_silver)
        else:
            X_img_silver = csr_matrix((0, 2048))

        if img_gold.size > 0:
            X_img_gold = csr_matrix(img_gold)
        else:
            X_img_gold = csr_matrix((0, 2048))

        # Log results
        log.info(f"   📊 Image Processing Results:")
        log.info(f"   ├─ Silver images available: {len(silver_img):,}")
        log.info(f"   ├─ Silver images downloaded: {len(silver_downloaded):,}")
        log.info(f"   ├─ Gold images available: {len(gold_img):,}")
        log.info(f"   ├─ Gold images downloaded: {len(gold_downloaded):,}")
        log.info(f"   ├─ Silver embeddings: {img_silver.shape}")
        log.info(f"   ├─ Gold embeddings: {img_gold.shape}")
        log.info(f"   └─ Embedding size: {img_silver.nbytes // (1024**2) if img_silver.size > 0 else 0} MB")

        # Early exit check for image-only mode
        if mode == "image" and (img_silver.size == 0 or img_gold.size == 0):
            log.warning(f"   ⚠️  Image-only mode requested but no valid images available!")
            log.warning(f"   └─ Consider using mode='text' or increasing sample_frac")
            stage_time = time.time() - stage_start
            log.info(f"   ❌ Image processing failed in {stage_time:.1f}s")
            return None, None, None, []

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
    if mode in {"image", "both"} and img_silver and img_silver.size > 0:
        training_subtasks.append("Image Models")
    if mode in {"text", "both"}:
        training_subtasks.append("Text Models")
    if mode == "both" and img_silver and img_silver.size > 0:
        training_subtasks.append("Text+Image Ensemble")
        training_subtasks.append("Final Combined")

    with tqdm(training_subtasks, desc="   ├─ Training Phases", position=1, leave=False,
              bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as train_pbar:

        # IMAGE MODELS
        if mode in {"image", "both"} and img_silver and img_silver.size > 0:
            train_pbar.set_description("   ├─ Training Image Models")
            log.info(f"   🖼️  Training image-based models...")

            res_img = run_mode_A(
                X_img_silver,
                img_gold_df.clean,
                X_img_gold,
                img_silver_df,
                img_gold_df,
                domain="image",
                apply_smote=False
            )

            results.extend(res_img)
            log.info(f"      ✅ Image models: {len(res_img)} results")
            optimize_memory_usage("Image Models")
            train_pbar.update(1)

        # TEXT MODELS
        if mode in {"text", "both"}:
            train_pbar.set_description("   ├─ Training Text Models")
            log.info(f"   🔤 Training text-based models...")

            res_text = run_mode_A(
                X_text_silver, gold.clean, X_text_gold,
                silver_txt, gold,
                domain="text", apply_smote=True
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
            for task in ("keto", "vegan"):
                try:
                    # Find best models
                    text_models = [r for r in res_text if r["task"] == task]
                    image_models = [r for r in res_img if r["task"] == task]

                    if not text_models or not image_models:
                        log.warning(f"      ⚠️  No models available for {task} ensemble")
                        continue

                    bt = max(text_models, key=lambda r: r["F1"])
                    bi = max(image_models, key=lambda r: r["F1"])

                    log.info(f"      ├─ {task}: Text={bt['model']} (F1={bt['F1']:.3f}), "
                             f"Image={bi['model']} (F1={bi['F1']:.3f})")

                    # Align predictions
                    if len(bt["prob"]) == len(gold) and len(bi["prob"]) == len(img_gold_df):
                        s_txt = pd.Series(bt["prob"], index=gold.index)
                        s_img = pd.Series(bi["prob"], index=img_gold_df.index)
                        common = s_txt.index.intersection(s_img.index)

                        log.info(f"      ├─ Alignment: {len(s_txt)} text + {len(s_img)} image = {len(common)} common")

                        if len(common) >= 10:
                            # Average predictions
                            avg = (s_txt.loc[common] + s_img.loc[common]) / 2

                            ensemble_result = pack(gold.loc[common, f"label_{task}"].values, avg.values) | {
                                "model": "TxtImg", "task": task,
                                "prob": avg.values, "pred": (avg.values >= .5).astype(int),
                                "text_model": bt['model'],
                                "image_model": bi['model'],
                                "common_samples": len(common)
                            }
                            ensemble_results.append(ensemble_result)

                            log.info(f"      ✅ {task} ensemble: F1={ensemble_result['F1']:.3f}")
                        else:
                            log.warning(f"      ⚠️  Too few common samples ({len(common)}) for {task} ensemble")
                    else:
                        log.warning(f"      ⚠️  Dimension mismatch for {task}: text={len(bt['prob'])}, image={len(bi['prob'])}")

                except Exception as e:
                    log.error(f"      ❌ {task} ensemble creation failed: {str(e)[:50]}...")

            if ensemble_results:
                table("Ensemble Text+Image", ensemble_results)
                results.extend(ensemble_results)
                log.info(f"      ✅ Created {len(ensemble_results)} ensembles")
            else:
                log.warning(f"      ⚠️  No successful ensembles created")

            train_pbar.update(1)

        # FINAL COMBINED MODEL TRAINING
        if mode == "both" and img_silver and img_silver.size > 0:
            train_pbar.set_description("   ├─ Final Combined Models")
            log.info(f"   🔄 Training final combined models...")

            # Align features and data
            common_silver_idx = img_silver_df.index
            common_gold_idx = img_gold_df.index

            if len(common_silver_idx) > 0 and len(common_gold_idx) > 0:
                # Align silver features
                X_text_silver_algn = vec.transform(silver_txt.loc[common_silver_idx].clean)
                X_silver = combine_features(X_text_silver_algn, img_silver)

                # Align gold features
                X_text_gold_algn = vec.transform(gold.loc[common_gold_idx].clean)
                X_gold = combine_features(X_text_gold_algn, img_gold)

                silver_eval = silver_txt.loc[common_silver_idx]
                gold_eval = gold.loc[common_gold_idx]

                log.info(f"      ├─ Combined silver features: {X_silver.shape}")
                log.info(f"      ├─ Combined gold features: {X_gold.shape}")
                log.info(f"      ├─ Silver samples: {len(silver_eval):,}")
                log.info(f"      └─ Gold samples: {len(gold_eval):,}")

                # Run combined training
                res_combined = run_mode_A(
                    X_silver, gold_eval.clean, X_gold,
                    silver_eval, gold_eval,
                    domain="both", apply_smote=True
                )
                results.extend(res_combined)
                log.info(f"      ✅ Combined models: {len(res_combined)} results")
                optimize_memory_usage()

            else:
                log.warning(f"      ⚠️  No common indices for combined features, skipping")

            train_pbar.update(1)

        # Setup feature matrices for ensemble creation
        if mode == "both" and img_silver and img_silver.size > 0:
            X_silver, X_gold = X_silver, X_gold
            silver_eval = silver_eval
        elif mode == "text":
            X_silver, X_gold = X_text_silver, X_text_gold
            silver_eval = silver_txt
        elif mode == "image" and img_silver and img_silver.size > 0:
            X_silver, X_gold = csr_matrix(img_silver), csr_matrix(img_gold)
            silver_eval = img_silver_df
        else:
            # Fallback to text
            log.warning(f"   ⚠️  No valid images for image mode, falling back to text")
            X_silver, X_gold = X_text_silver, X_text_gold
            silver_eval = silver_txt

        # Final training if no results yet
        if not results:
            log.info(f"   🎯 Running fallback text-only training...")
            res_final = run_mode_A(X_text_silver, gold.clean, X_text_gold,
                                   silver_txt, gold, domain="text", apply_smote=True)
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
                task_models = [r for r in results if r["task"] == task and r["model"] != "Rule"]
                log.info(f"      ├─ Available models: {len(task_models)}")

                if len(task_models) > 1:
                    # Use appropriate features
                    if mode == "both" and img_silver and img_silver.size > 0:
                        ens_X_silver = X_silver
                        ens_X_gold = X_gold
                        ens_silver_eval = silver_eval
                    elif mode == "image" and img_silver and img_silver.size > 0:
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
                        log.info(f"      ✅ {task} ensemble: {best_ens['model']} (F1={best_ens['F1']:.3f})")
                    else:
                        log.warning(f"      ⚠️  {task} ensemble optimization failed")
                else:
                    log.info(f"      ⏭️  {task}: Only {len(task_models)} model(s) available, skipping ensemble")

            results.extend(ensemble_results)
            log.info(f"   📊 Ensemble results: {len(ensemble_results)} optimized ensembles")
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
            pd.DataFrame(results_summary).to_csv("pipeline_results_summary.csv", index=False)
            log.info(f"      ✅ Saved results summary with {len(results_summary)} entries")
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
    log.info(f"   ├─ Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
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
    else:
        log.warning(f"\n   ⚠️  NO RESULTS GENERATED")
        log.warning(f"   └─ Consider checking data availability or adjusting parameters")

    # Resource usage summary
    final_memory = psutil.virtual_memory()
    log.info(f"\n   💾 RESOURCE USAGE:")
    log.info(f"   ├─ Peak memory: {final_memory.percent:.1f}%")
    log.info(f"   ├─ Final memory: {final_memory.used // (1024**2)} MB")
    log.info(f"   └─ Efficiency: {len(results)/total_time:.2f} models/second")

    # Save pipeline metadata
    pipeline_metadata = {
        'mode': mode,
        'force': force,
        'sample_frac': sample_frac,
        'total_time': total_time,
        'total_results': len(results),
        'start_time': pipeline_start,
        'end_time': time.time(),
        'memory_peak_percent': final_memory.percent,
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total // (1024**3)
        },
        'data_stats': {
            'silver_text_size': len(silver_txt),
            'silver_image_size': len(silver_img),
            'gold_size': len(gold),
            'silver_images_downloaded': len(silver_downloaded) if 'silver_downloaded' in locals() else 0,
            'gold_images_downloaded': len(gold_downloaded) if 'gold_downloaded' in locals() else 0
        }
    }

    with open("pipeline_metadata.json", "w") as f:
        json.dump(pipeline_metadata, f, indent=2)

    log.info(f"   💾 Saved pipeline metadata to pipeline_metadata.json")

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
    
    Supports multiple modes:
    - Training: Run full pipeline to train models
    - Inference: Classify ingredients using trained models
    - Evaluation: Test on ground truth dataset
    
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

    args = parser.parse_args()

    try:
        log.info(f"🚀 Starting main with args: {args}")

        if args.ingredients:
            # Handle ingredient classification
            if args.ingredients.startswith('['):
                ingredients = json.loads(args.ingredients)
            else:
                ingredients = [i.strip() for i in args.ingredients.split(',') if i.strip()]

            keto = is_keto(ingredients)
            vegan = is_vegan(ingredients)
            print(json.dumps({'keto': keto, 'vegan': vegan}))
            return

        elif args.train:
            log.info(f"🧠 SINGLE training run - sample_frac={args.sample_frac}")

            try:
                vec, silver, gold, res = run_full_pipeline(
                    mode=args.mode, force=args.force, sample_frac=args.sample_frac)

                if not res:
                    log.error("❌ Pipeline produced no results!")
                    sys.exit(1)

                log.info(f"✅ Pipeline completed with {len(res)} results")

                # Save models
                try:
                    import pickle
                    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)

                    # Save vectorizer
                    with open(CFG.artifacts_dir / "vectorizer.pkl", 'wb') as f:
                        pickle.dump(vec, f)
                    log.info("✅ Saved vectorizer")

                    # Save best models
                    best_models = {}
                    for task in ['keto', 'vegan']:
                        task_res = [r for r in res if r['task'] == task]
                        if task_res:
                            best = max(task_res, key=lambda x: x['F1'])
                            model_name = best['model']
                            base_name = model_name.split('_')[0]

                            if base_name in BEST:
                                best_models[task] = BEST[base_name]
                                log.info(f"✅ Saved {task} model: {base_name}")
                            else:
                                log.warning(f"⚠️  Could not find model {base_name} in BEST dict")

                    if best_models:
                        with open(CFG.artifacts_dir / "models.pkl", 'wb') as f:
                            pickle.dump(best_models, f)
                        log.info(f"✅ Saved {len(best_models)} models to {CFG.artifacts_dir}")
                    else:
                        log.warning("⚠️  No models to save")

                except Exception as e:
                    log.error(f"❌ Could not save models: {e}")

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
                    log.error(f"❌ Ground truth file not found: {args.ground_truth}")
                    log.info("💡 Make sure to run the command with:")
                    log.info("   python diet_classifiers.py --ground_truth /usr/src/data/ground_truth_sample.csv")
                    sys.exit(1)

                df = pd.read_csv(args.ground_truth)
                log.info(f"✅ Loaded ground truth with {len(df)} rows")

                # Resolve model/vectorizer paths with fallback logic
                model_path = CFG.artifacts_dir / "models.pkl"
                vec_path = CFG.artifacts_dir / "vectorizer.pkl"

                if not (model_path.exists() and vec_path.exists()):
                    log.warning("⚠️ Models not found in artifacts/ — trying pretrained_models/ fallback...")
                    model_path = CFG.pretrained_models_dir / "models.pkl"
                    vec_path = CFG.pretrained_models_dir / "vectorizer.pkl"

                if not (model_path.exists() and vec_path.exists()):
                    log.error("❌ No trained models found in artifacts/ or pretrained_models/.")
                    sys.exit(1)

                with open(vec_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                log.info(f"✅ Loaded vectorizer from {vec_path}")

                with open(model_path, 'rb') as f:
                    models = pickle.load(f)
                log.info(f"✅ Loaded models from {model_path}")

                # Vectorize ingredients
                texts = df['ingredients'].fillna("").tolist()
                X = vectorizer.transform(texts)
                log.info("✅ Transformed text to feature vectors")

                # Predict
                results = []
                for idx, row in df.iterrows():
                    res = {"index": idx}
                    for task in ["keto", "vegan"]:
                        if task in models:
                            pred = models[task].predict(X[idx])
                            res[f"{task}_pred"] = int(pred[0])
                        else:
                            res[f"{task}_pred"] = None
                            log.warning(f"⚠️ No trained model found for task: {task}")
                    res["keto_true"] = row.get("label_keto")
                    res["vegan_true"] = row.get("label_vegan")
                    results.append(res)

                results_df = pd.DataFrame(results)

                # Save predictions
                pred_path = CFG.artifacts_dir / "ground_truth_predictions.csv"
                results_df.to_csv(pred_path, index=False)
                log.info(f"✅ Saved predictions to {pred_path}")

                # Evaluation metrics
                metrics = []
                for task in ["keto", "vegan"]:
                    true_col = f"{task}_true"
                    pred_col = f"{task}_pred"

                    if true_col in results_df.columns:
                        y_true = results_df[true_col].dropna().astype(int)
                        y_pred = results_df[pred_col].dropna().astype(int)

                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)

                        metrics.append({
                            "task": task,
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1_score": f1
                        })

                        log.info(f"✔️  {task.capitalize()} - ACC: {acc:.3f} | PREC: {prec:.3f} | REC: {rec:.3f} | F1: {f1:.3f}")

                # Save metrics
                metrics_df = pd.DataFrame(metrics)
                metrics_path = CFG.artifacts_dir / "eval_metrics.csv"
                metrics_df.to_csv(metrics_path, index=False)
                log.info(f"📈 Saved evaluation metrics to {metrics_path}")

            except Exception as e:
                log.error(f"❌ Ground truth evaluation failed: {e}")
                sys.exit(1)


            except Exception as e:
                log.error(f"❌ Ground truth evaluation failed: {e}")
                sys.exit(1)

        else:
            # Default pipeline
            log.info(f"🧠 Default pipeline - sample_frac={args.sample_frac}")

            try:
                run_full_pipeline(mode=args.mode, force=args.force,
                                  sample_frac=args.sample_frac)
            except Exception as e:
                log.error(f"❌ Default pipeline failed: {e}")
                sys.exit(1)

        log.info("🏁 Main completed successfully")

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
        print(f"❌ RESTART LOOP DETECTED (count={restart_count}) - STOPPING")
        sys.exit(1)

    # Set restart counter
    os.environ['PIPELINE_RESTART_COUNT'] = str(restart_count + 1)

    try:
        main()
    except Exception as e:
        print(f"❌ Final exception caught: {e}")
        sys.exit(1)
    finally:
        # Clear restart counter on normal exit
        if 'PIPELINE_RESTART_COUNT' in os.environ:
            del os.environ['PIPELINE_RESTART_COUNT']
