#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading functions for the diet classification pipeline.

"""

from pathlib import Path
from typing import Tuple, Optional
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil

from ..core import log, get_pipeline_state
from ..config import CFG
from .preprocessing import normalise
from .silver_labels import build_silver
from .usda import _load_usda_carb_table, label_usda_keto_data


def get_datasets(sample_frac: Optional[float] = None
                 ) -> Tuple[pd.DataFrame, pd.DataFrame,
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
        
    Based on original lines 752-779
    """
    pipeline_state = get_pipeline_state()

    if pipeline_state.datasets is None:
        # Actually load everything once
        silver_all, gold, recipes, carb_df = load_datasets()

        # Optional row-sampling of the *silver* set
        if sample_frac:
            silver_all = silver_all.sample(frac=sample_frac,
                                           random_state=42).copy()

        pipeline_state.datasets = (silver_all, gold, recipes, carb_df)

    return pipeline_state.datasets


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        
    Based on original lines 1068-1368 (trimmed for brevity - this is a 300+ line function)
    """
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

    # Validate and load recipes
    try:
        recipes = pd.read_parquet(recipes_url)
    except Exception as e:
        log.error(f"      ❌ Failed to load recipes: {str(e)[:100]}...")
        raise RuntimeError(f"Failed to load recipes: {e}")

    log.info(f"      📊 Recipes Dataset:")
    log.info(f"      ├─ Shape: {recipes.shape}")
    log.info(f"      ├─ Columns: {list(recipes.columns)}")
    log.info(
        f"      └─ Memory usage: {recipes.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

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

    # Load ground truth
    try:
        ground_truth = pd.read_csv(gt_url_or_path)
    except UnicodeDecodeError:
        log.warning(f"      ⚠️  UTF-8 decode failed, trying latin-1...")
        ground_truth = pd.read_csv(gt_url_or_path, encoding='latin-1')
    except pd.errors.EmptyDataError:
        log.error(f"      ❌ Ground truth file is empty")
        raise RuntimeError("Ground truth CSV file is empty")

    # Process labels
    keto_columns = [col for col in ground_truth.columns if 'keto' in col.lower()]
    vegan_columns = [col for col in ground_truth.columns if 'vegan' in col.lower()]

    if keto_columns:
        ground_truth["label_keto"] = ground_truth.filter(
            regex="keto").iloc[:, 0].astype(int)
    else:
        log.warning(f"      ⚠️  No keto columns found - creating dummy labels")
        ground_truth["label_keto"] = 0

    if vegan_columns:
        ground_truth["label_vegan"] = ground_truth.filter(
            regex="vegan").iloc[:, 0].astype(int)
    else:
        log.warning(f"      ⚠️  No vegan columns found - creating dummy labels")
        ground_truth["label_vegan"] = 0

    # Add photo_url if available
    ground_truth["photo_url"] = ground_truth.get("photo_url")

    # Clean ingredients text
    ground_truth["clean"] = ground_truth.ingredients.fillna("").map(normalise)

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

    silver = build_silver(recipes)

    # Add photo URLs from recipes
    silver["photo_url"] = recipes.get("photo_url")

    # Calculate silver label statistics
    silver_stats = {
        'total': len(silver),
        'keto_positive': silver['silver_keto'].sum() if 'silver_keto' in silver.columns else 0,
        'vegan_positive': silver['silver_vegan'].sum() if 'silver_vegan' in silver.columns else 0,
        'has_photos': (~silver['photo_url'].isnull()).sum() if 'photo_url' in silver.columns else 0
    }

    log.info(f"      📊 Silver Labels Generated:")
    log.info(f"      ├─ Total recipes: {silver_stats['total']:,}")
    log.info(
        f"      ├─ Keto positive: {silver_stats['keto_positive']:,} ({silver_stats['keto_positive']/silver_stats['total']*100:.1f}%)")
    log.info(
        f"      ├─ Vegan positive: {silver_stats['vegan_positive']:,} ({silver_stats['vegan_positive']/silver_stats['total']*100:.1f}%)")
    log.info(
        f"      ├─ With photos: {silver_stats['has_photos']:,} ({silver_stats['has_photos']/silver_stats['total']*100:.1f}%)")

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

    # Check index alignment between datasets
    recipes_indices = set(recipes.index)
    silver_indices = set(silver.index)
    gt_indices = set(ground_truth.index)

    if recipes_indices != silver_indices:
        log.warning(f"      ⚠️  Index mismatch between recipes and silver")
        log.info(f"         ├─ Recipes: {len(recipes_indices)} indices")
        log.info(f"         └─ Silver: {len(silver_indices)} indices")

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


def show_balance(df: pd.DataFrame, title: str) -> None:
    """
    Print class distribution statistics.

    Displays the positive/negative class balance for both keto and vegan
    labels in a formatted table.

    Args:
        df: DataFrame containing label columns
        title: Title for the display
        
    Based on original lines 3566-3585
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