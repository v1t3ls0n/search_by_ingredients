#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline orchestration for diet classification.

Based on original lines 7268-8509 from diet_classifiers.py
This is a stub implementation - the full implementation would be ~1200 lines.
"""

import time
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd

from ..core import log, get_pipeline_state
from ..config import CFG
from ..data.loaders import get_datasets
from ..utils.memory import optimize_memory_usage


def run_full_pipeline(mode: str = "both",
                      force: bool = False,
                      sample_frac: Optional[float] = None) -> Tuple[Any, pd.DataFrame, pd.DataFrame, List[Dict]]:
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
    
    Based on original lines 7268-8509 (This is a simplified stub)
    """
    # Initialize pipeline tracking
    pipeline_start = time.time()
    pipeline_state = get_pipeline_state()

    # Log pipeline initialization
    log.info("🚀 STARTING FULL ML PIPELINE")
    log.info(f"   Mode: {mode}")
    log.info(f"   Force recomputation: {force}")
    log.info(f"   Sample fraction: {sample_frac or 'Full dataset'}")

    # Memory usage tracking
    def log_memory_usage(stage: str):
        import psutil
        memory = psutil.virtual_memory()
        log.info(f"   📊 {stage} - Memory: {memory.percent:.1f}% used")

    # ------------------------------------------------------------------
    # 1. DATA LOADING AND PREPARATION
    # ------------------------------------------------------------------
    log.info("\n📂 STAGE 1: DATA LOADING AND PREPARATION")
    
    silver_all, gold, recipes, carb_df = get_datasets(sample_frac)
    
    # Add UIDs
    silver_all["uid"] = silver_all.index
    gold["uid"] = gold.index
    
    # Show class balance
    from ..data.loaders import show_balance
    show_balance(gold, "Gold set")
    show_balance(silver_all, "Silver set")
    
    log_memory_usage("Data Loading")
    optimize_memory_usage("Data Loading")

    # ------------------------------------------------------------------
    # 2. TEXT FEATURE PROCESSING
    # ------------------------------------------------------------------
    log.info("\n🔤 STAGE 2: TEXT FEATURE PROCESSING")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        log.error("scikit-learn required for text features")
        return None, silver_all, gold, []
    
    vec = TfidfVectorizer(**CFG.vec_kwargs)
    
    # Prepare text data
    silver_txt = silver_all.copy()
    
    # Add USDA data to silver
    from ..data.usda import label_usda_keto_data
    if not carb_df.empty:
        usda_labeled = label_usda_keto_data(carb_df)
        log.info(f"   ├─ USDA examples added: {len(usda_labeled)}")
        silver_txt = pd.concat([silver_txt, usda_labeled], ignore_index=True)
    
    # Fit vectorizer
    X_text_silver = vec.fit_transform(silver_txt.clean)
    X_text_gold = vec.transform(gold.clean)
    
    log.info(f"   📊 Text Processing Results:")
    log.info(f"   ├─ Vocabulary size: {len(vec.vocabulary_):,}")
    log.info(f"   ├─ Silver features: {X_text_silver.shape}")
    log.info(f"   └─ Gold features: {X_text_gold.shape}")
    
    log_memory_usage("Text Processing")
    optimize_memory_usage("Text Processing")

    # ------------------------------------------------------------------
    # 3. IMAGE FEATURE PROCESSING (if requested)
    # ------------------------------------------------------------------
    if mode in {"image", "both"}:
        log.info("\n🖼️  STAGE 3: IMAGE FEATURE PROCESSING")
        log.warning("Image processing not implemented in this stub")
        # Full implementation would download images and extract embeddings
        X_img_silver = None
        X_img_gold = None
    else:
        log.info("\n⏭️  STAGE 3: SKIPPED (Image processing not requested)")

    # ------------------------------------------------------------------
    # 4. MODEL TRAINING
    # ------------------------------------------------------------------
    log.info("\n🤖 STAGE 4: MODEL TRAINING")
    
    results = []
    
    if mode in {"text", "both"}:
        log.info(f"   🔤 Training text-based models...")
        
        # In full implementation, this would call run_mode_A
        # For now, just train a simple model
        from ..models.builders import build_models
        from ..evaluation.metrics import pack
        
        for task in ["keto", "vegan"]:
            log.info(f"   Training {task} models...")
            
            y_train = silver_txt[f"silver_{task}"].values
            y_test = gold[f"label_{task}"].values
            
            models = build_models(task, "text")
            
            for model_name, model in models.items():
                try:
                    log.info(f"      Training {model_name}...")
                    model.fit(X_text_silver, y_train)
                    
                    # Predict
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_text_gold)[:, 1]
                    else:
                        prob = model.decision_function(X_text_gold)
                        prob = 1 / (1 + np.exp(-prob))  # Sigmoid
                    
                    # Calculate metrics
                    metrics = pack(y_test, prob)
                    
                    result = {
                        'task': task,
                        'model': f"{model_name}_TEXT",
                        'domain': 'text',
                        **metrics,
                        'prob': prob,
                        'pred': (prob >= 0.5).astype(int)
                    }
                    
                    results.append(result)
                    
                    log.info(f"      ✅ {model_name}: F1={metrics['F1']:.3f}")
                    
                except Exception as e:
                    log.error(f"      ❌ {model_name} failed: {e}")
        
        # Display results
        from ..evaluation.metrics import table
        table("Text Models", results)
    
    log_memory_usage("Model Training")

    # ------------------------------------------------------------------
    # 5. ENSEMBLE OPTIMIZATION
    # ------------------------------------------------------------------
    if len(results) > 1:
        log.info("\n🎭 STAGE 5: ENSEMBLE OPTIMIZATION")
        log.warning("Ensemble optimization not implemented in this stub")
    
    # ------------------------------------------------------------------
    # 6. EVALUATION AND EXPORT  
    # ------------------------------------------------------------------
    log.info("\n📊 STAGE 6: EVALUATION AND EXPORT")
    
    # Save best models
    if results:
        from ..models.io import save_models_optimized
        
        best_models = {}
        for task in ['keto', 'vegan']:
            task_results = [r for r in results if r['task'] == task]
            if task_results:
                best = max(task_results, key=lambda x: x['F1'])
                # In full implementation, would get actual model object
                # For stub, just use a placeholder
                from ..models.rules import RuleModel
                best_models[task] = RuleModel(task)
        
        if best_models:
            save_models_optimized(best_models, vec, CFG.artifacts_dir)
            log.info("   ✅ Models saved")
    
    # ------------------------------------------------------------------
    # PIPELINE COMPLETION
    # ------------------------------------------------------------------
    total_time = time.time() - pipeline_start
    
    log.info(f"\n🏁 PIPELINE COMPLETE")
    log.info(f"   ├─ Total runtime: {total_time:.1f}s")
    log.info(f"   ├─ Mode: {mode}")
    log.info(f"   └─ Total results: {len(results)}")
    
    # Update pipeline state
    pipeline_state.vectorizer = vec
    pipeline_state.models = best_models if 'best_models' in locals() else {}
    pipeline_state.initialized = True
    
    return vec, silver_txt, gold, results





# ------------------------- Full Function Source Reference From Singleton Code Version -----------------------------------------

# def run_full_pipeline(mode: str = "both",
#                       force: bool = False,
#                       sample_frac: float | None = None):
#     """
#     Execute the complete machine learning pipeline for diet classification.

#     This is the main orchestration function that:
#     1. Loads all datasets (recipes, ground truth, USDA)
#     2. Generates silver labels for training
#     3. Extracts text features (TF-IDF)
#     4. Downloads images and extracts visual features (ResNet-50)
#     5. Trains multiple models on different feature types
#     6. Creates optimized ensembles
#     7. Evaluates on gold standard test set
#     8. Exports results and visualizations

#     Args:
#         mode: Feature mode - 'text', 'image', or 'both'
#         force: Force recomputation of cached embeddings
#         sample_frac: Fraction of silver data to sample (for testing)

#     Returns:
#         Tuple of (vectorizer, silver_data, gold_data, results)

#     The function includes comprehensive error handling, memory optimization,
#     and detailed progress tracking throughout all stages.
#     """

#     # Initialize pipeline tracking
#     pipeline_start = time.time()

#     # Log pipeline initialization
#     log.info("🚀 STARTING FULL ML PIPELINE")
#     log.info(f"   Mode: {mode}")
#     log.info(f"   Force recomputation: {force}")
#     log.info(f"   Sample fraction: {sample_frac or 'Full dataset'}")
#     log.info(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     log.info(f"   Available CPU cores: {psutil.cpu_count()}")
#     log.info(
#         f"   Available memory: {psutil.virtual_memory().total // (1024**3)} GB")

#     # Memory usage tracking
#     def log_memory_usage(stage: str):
#         memory = psutil.virtual_memory()
#         log.info(f"   📊 {stage} - Memory: {memory.percent:.1f}% used "
#                  f"({memory.used // (1024**2)} MB / {memory.total // (1024**2)} MB)")

#     # Pipeline stages
#     pipeline_stages = [
#         "Data Loading", "Text Processing", "Image Processing",
#         "Model Training", "Ensemble Creation", "Evaluation"
#     ]

#     # Main progress bar
#     pipeline_progress = tqdm(pipeline_stages, desc="🔬 ML Pipeline",
#                              position=0, leave=True,
#                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}")

#     # ------------------------------------------------------------------
#     # 1. DATA LOADING AND PREPARATION
#     # ------------------------------------------------------------------
#     pipeline_progress.set_description("🔬 ML Pipeline: Data Loading")
#     stage_start = time.time()

#     log.info("\n📂 STAGE 1: DATA LOADING AND PREPARATION")

#     with tqdm(total=4, desc="   ├─ Loading Data", position=1, leave=False,
#               bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as load_pbar:

#         load_pbar.set_description("   ├─ Loading datasets")
#         silver_all, gold, _, _ = get_datasets(sample_frac)
#         load_pbar.update(1)

#         load_pbar.set_description("   ├─ Creating index keys")
#         silver_all["uid"] = silver_all.index
#         gold["uid"] = gold.index
#         load_pbar.update(1)

#         load_pbar.set_description("   ├─ Preparing text data")
#         silver_txt = silver_all.copy()
#         load_pbar.update(1)

#         load_pbar.set_description("   ├─ Filtering image data")
#         silver_img = filter_photo_rows(silver_all)
#         gold_img = filter_photo_rows(gold)
#         load_pbar.update(1)

#     # Apply sampling if requested
#     if sample_frac and sample_frac < 1.0:
#         original_txt_size = len(silver_txt)
#         original_img_size = len(silver_img)

#         # Sample both datasets consistently
#         silver_txt = silver_txt.sample(
#             frac=sample_frac, random_state=42).copy()

#         if not silver_img.empty:
#             # Get consistent sampling across modalities
#             sampled_indices = silver_txt.index
#             available_img_indices = silver_img.index
#             common_indices = sampled_indices.intersection(
#                 available_img_indices)

#             if len(common_indices) > 0:
#                 silver_img = silver_img.loc[common_indices].copy()
#                 log.info(
#                     f"   📉 Consistent sampling: Using {len(common_indices):,} common indices")
#             else:
#                 silver_img = silver_img.sample(
#                     frac=sample_frac, random_state=42).copy()
#                 log.info(f"   📉 Separate sampling: No common indices found")

#         sampled_txt_size = len(silver_txt)
#         sampled_img_size = len(silver_img)

#         log.info(f"   📉 Applied sampling before processing:")
#         log.info(
#             f"   ├─ Text: {original_txt_size:,} → {sampled_txt_size:,} rows ({sample_frac:.1%})")
#         log.info(
#             f"   └─ Images: {original_img_size:,} → {sampled_img_size:,} rows ({sample_frac:.1%})")

#     # Log dataset statistics
#     log.info(f"\n   📊 Dataset Statistics:")
#     log.info(f"   ├─ Silver (All): {len(silver_all):,} recipes")
#     log.info(f"   ├─ Silver (Text): {len(silver_txt):,} recipes")
#     log.info(f"   ├─ Silver (Images): {len(silver_img):,} recipes")
#     log.info(f"   ├─ Gold (All): {len(gold):,} recipes")
#     log.info(f"   └─ Gold (Images): {len(gold_img):,} recipes")

#     # Display class balance
#     log.info(f"\n   ⚖️  Class Balance Analysis:")
#     show_balance(gold, "Gold set")
#     show_balance(silver_txt, "Silver (Text) set")
#     show_balance(silver_img, "Silver (Image) set")

#     stage_time = time.time() - stage_start
#     log.info(f"   ✅ Data loading completed in {stage_time:.1f}s")
#     log_memory_usage("Data Loading")
#     pipeline_progress.update(1)
#     optimize_memory_usage("Data Loading")

#     # Memory check
#     if psutil.virtual_memory().percent > 70:
#         log.warning(
#             f"High memory usage after data loading: {psutil.virtual_memory().percent:.1f}%")

#     # ------------------------------------------------------------------
#     # 2. TEXT FEATURE PROCESSING
#     # ------------------------------------------------------------------
#     pipeline_progress.set_description("🔬 ML Pipeline: Text Processing")
#     stage_start = time.time()

#     log.info("\n🔤 STAGE 2: TEXT FEATURE PROCESSING")

#     with tqdm(total=4, desc="   ├─ Text Features", position=1, leave=False,
#               bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as text_pbar:

#         text_pbar.set_description("   ├─ Initializing vectorizer")
#         vec = TfidfVectorizer(**CFG.vec_kwargs)
#         log.info(f"   ├─ Vectorizer config: {CFG.vec_kwargs}")
#         text_pbar.update(1)

#         text_pbar.set_description("   ├─ Fitting on silver data")

#         # Load USDA carb data and convert to keto-labeled rows
#         carb_df = _load_usda_carb_table()
#         if not carb_df.empty:
#             usda_labeled = label_usda_keto_data(carb_df)
#             log.info(f"   ├─ USDA examples added: {len(usda_labeled)}")
#             silver_txt = pd.concat(
#                 [silver_txt, usda_labeled], ignore_index=True)
#             Path("artifacts").mkdir(exist_ok=True)
#             silver_txt.to_csv("artifacts/silver_extended.csv", index=False)
#         else:
#             log.warning("   ├─ No USDA data added - carb_df is empty")

#         X_text_silver = vec.fit_transform(silver_txt.clean)
#         text_pbar.update(1)

#         text_pbar.set_description("   ├─ Transforming gold data")
#         X_text_gold = vec.transform(gold.clean)
#         text_pbar.update(1)

#         text_pbar.set_description("   ├─ Saving embeddings")
#         Path("embeddings").mkdir(exist_ok=True)
#         joblib.dump(X_text_gold, "embeddings/text_gold.pkl")
#         text_pbar.update(1)

#     # Log text processing results
#     log.info(f"   📊 Text Processing Results:")
#     log.info(f"   ├─ Vocabulary size: {len(vec.vocabulary_):,}")
#     log.info(f"   ├─ Silver features: {X_text_silver.shape}")
#     log.info(f"   ├─ Gold features: {X_text_gold.shape}")
#     log.info(
#         f"   ├─ Sparsity: {(1 - X_text_silver.nnz / X_text_silver.size):.1%}")
#     log.info(
#         f"   └─ Memory usage: ~{X_text_silver.data.nbytes // (1024**2)} MB")

#     stage_time = time.time() - stage_start
#     log.info(f"   ✅ Text processing completed in {stage_time:.1f}s")
#     log_memory_usage("Text Processing")
#     optimize_memory_usage("Text Processing")
#     pipeline_progress.update(1)

#     # Initialize result containers
#     results, res_text, res_img = [], [], []
#     img_silver = img_gold = None

#     # ------------------------------------------------------------------
#     # 3. IMAGE FEATURE PROCESSING
#     # ------------------------------------------------------------------
#     if mode in {"image", "both"}:
#         pipeline_progress.set_description("🔬 ML Pipeline: Image Processing")
#         stage_start = time.time()

#         log.info("\n🖼️  STAGE 3: IMAGE FEATURE PROCESSING")
#         log.info(f"   ├─ Processing {len(silver_img):,} sampled silver images")
#         log.info(f"   └─ Processing {len(gold_img):,} gold images")

#         with tqdm(total=6, desc="   ├─ Image Pipeline", position=1, leave=False,
#                   bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as img_pbar:

#             # Download images with robust function and force parameter
#             img_pbar.set_description("   ├─ Downloading silver images")
#             if not silver_img.empty:
#                 silver_downloaded = _download_images(
#                     silver_img, CFG.image_dir / "silver", max_workers=16, force=force)
#                 log.info(
#                     f"      ├─ Silver download: {len(silver_downloaded):,}/{len(silver_img):,} successful")
#             else:
#                 silver_downloaded = []
#                 log.info(f"      ├─ Silver download: No images to download")
#             img_pbar.update(1)

#             img_pbar.set_description("   ├─ Downloading gold images")
#             if not gold_img.empty:
#                 gold_downloaded = _download_images(
#                     gold_img, CFG.image_dir / "gold", max_workers=16, force=force)
#                 log.info(
#                     f"      ├─ Gold download: {len(gold_downloaded):,}/{len(gold_img):,} successful")
#             else:
#                 gold_downloaded = []
#                 log.info(f"      ├─ Gold download: No images to download")
#             img_pbar.update(1)

#             # Enhanced filtering logic with better error handling
#             img_pbar.set_description("   ├─ Filtering by downloads")

#             # Silver image filtering
#             if silver_downloaded:
#                 try:
#                     # Use the indices that were actually downloaded
#                     img_silver_df = silver_img.loc[silver_img.index.intersection(
#                         silver_downloaded)].copy()

#                     if img_silver_df.empty:
#                         log.warning(
#                             f"      ⚠️  Silver filtering resulted in empty DataFrame")
#                         img_silver_df = pd.DataFrame()
#                     else:
#                         log.info(
#                             f"      ├─ Silver filtered: {len(img_silver_df):,} with valid images")

#                 except Exception as e:
#                     log.error(f"      ❌ Silver filtering failed: {e}")
#                     img_silver_df = pd.DataFrame()
#             else:
#                 img_silver_df = pd.DataFrame()
#                 log.info(f"      ├─ Silver filtered: Empty (no downloads)")

#             # Gold image filtering
#             if gold_downloaded:
#                 try:
#                     # Use the indices that were actually downloaded
#                     img_gold_df = gold_img.loc[gold_img.index.intersection(
#                         gold_downloaded)].copy()

#                     if img_gold_df.empty:
#                         log.warning(
#                             f"      ⚠️  Gold filtering resulted in empty DataFrame")
#                         img_gold_df = pd.DataFrame()
#                     else:
#                         log.info(
#                             f"      ├─ Gold filtered: {len(img_gold_df):,} with valid images")

#                 except Exception as e:
#                     log.error(f"      ❌ Gold filtering failed: {e}")
#                     img_gold_df = pd.DataFrame()
#             else:
#                 img_gold_df = pd.DataFrame()
#                 log.info(f"      ├─ Gold filtered: Empty (no downloads)")

#             # Gold image filtering
#             if gold_downloaded:
#                 try:
#                     img_gold_df = filter_photo_rows(gold_img)
#                     # Ensure we only keep images that were actually downloaded
#                     if not img_gold_df.empty:
#                         img_gold_df = img_gold_df.loc[img_gold_df.index.intersection(
#                             gold_downloaded)].copy()

#                     if img_gold_df.empty:
#                         log.warning(
#                             f"      ⚠️  Gold filtering resulted in empty DataFrame")
#                         # Fallback: use original indices from downloads
#                         img_gold_df = gold_img.loc[gold_downloaded].copy()

#                 except Exception as e:
#                     log.error(f"      ❌ Gold filtering failed: {e}")
#                     # Fallback to original gold_img subset
#                     img_gold_df = gold_img.loc[gold_downloaded].copy(
#                     ) if gold_downloaded else pd.DataFrame()
#             else:
#                 img_gold_df = pd.DataFrame()

#             log.info(
#                 f"      ├─ Gold filtered: {len(img_gold_df):,} with valid images")
#             img_pbar.update(1)

#             # Extract embeddings with proper alignment and error handling
#             img_pbar.set_description("   ├─ Building silver embeddings")
#             if not img_silver_df.empty:
#                 try:
#                     img_silver, silver_valid_indices = build_image_embeddings(
#                         img_silver_df, "silver", force)

#                     # Verify and align DataFrame to match embeddings
#                     if len(silver_valid_indices) != len(img_silver_df):
#                         log.info(
#                             f"      ├─ Aligning silver DataFrame: {len(img_silver_df):,} → {len(silver_valid_indices):,} rows")
#                         img_silver_df = img_silver_df.loc[silver_valid_indices].copy(
#                         )

#                     # Validate dimensions match
#                     if img_silver.shape[0] != len(img_silver_df):
#                         log.error(f"      ❌ Silver dimension mismatch after alignment: "
#                                   f"embeddings={img_silver.shape[0]}, df={len(img_silver_df)}")
#                         # Try to fix by truncating to smaller size
#                         min_size = min(img_silver.shape[0], len(img_silver_df))
#                         img_silver = img_silver[:min_size]
#                         img_silver_df = img_silver_df.iloc[:min_size].copy()
#                         log.info(
#                             f"      🔧 Truncated both to {min_size} for alignment")

#                     log.info(f"      ├─ Silver embeddings: {img_silver.shape}")
#                     log.info(
#                         f"      ├─ Silver DataFrame: {len(img_silver_df):,} rows")

#                 except Exception as e:
#                     log.error(
#                         f"      ❌ Silver embedding extraction failed: {e}")
#                     img_silver = np.array([]).reshape(0, 2048)
#                     img_silver_df = pd.DataFrame()
#                     silver_valid_indices = []
#                     log.info(f"      ├─ Silver fallback: Empty embeddings")
#             else:
#                 img_silver = np.array([]).reshape(0, 2048)
#                 silver_valid_indices = []
#                 log.info(
#                     f"      ├─ Silver embeddings: Empty array (no valid images)")
#             img_pbar.update(1)

#             img_pbar.set_description("   ├─ Building gold embeddings")
#             if not img_gold_df.empty:
#                 try:
#                     img_gold, gold_valid_indices = build_image_embeddings(
#                         img_gold_df, "gold", force)

#                     # Verify and align DataFrame to match embeddings
#                     if len(gold_valid_indices) != len(img_gold_df):
#                         log.info(
#                             f"      ├─ Aligning gold DataFrame: {len(img_gold_df):,} → {len(gold_valid_indices):,} rows")
#                         img_gold_df = img_gold_df.loc[gold_valid_indices].copy(
#                         )

#                     # Validate dimensions match
#                     if img_gold.shape[0] != len(img_gold_df):
#                         log.error(f"      ❌ Gold dimension mismatch after alignment: "
#                                   f"embeddings={img_gold.shape[0]}, df={len(img_gold_df)}")
#                         # Try to fix by truncating to smaller size
#                         min_size = min(img_gold.shape[0], len(img_gold_df))
#                         img_gold = img_gold[:min_size]
#                         img_gold_df = img_gold_df.iloc[:min_size].copy()
#                         log.info(
#                             f"      🔧 Truncated both to {min_size} for alignment")

#                     log.info(f"      ├─ Gold embeddings: {img_gold.shape}")
#                     log.info(
#                         f"      ├─ Gold DataFrame: {len(img_gold_df):,} rows")

#                 except Exception as e:
#                     log.error(f"      ❌ Gold embedding extraction failed: {e}")
#                     img_gold = np.array([]).reshape(0, 2048)
#                     img_gold_df = pd.DataFrame()
#                     gold_valid_indices = []
#                     log.info(f"      ├─ Gold fallback: Empty embeddings")
#             else:
#                 img_gold = np.array([]).reshape(0, 2048)
#                 gold_valid_indices = []
#                 log.info(
#                     f"      ├─ Gold embeddings: Empty array (no valid images)")
#             img_pbar.update(1)

#             img_pbar.set_description("   ├─ Saving embeddings")
#             try:
#                 # Ensure embeddings directory exists
#                 Path("embeddings").mkdir(exist_ok=True)

#                 if img_gold.size > 0:
#                     joblib.dump(img_gold, "embeddings/img_gold.pkl")
#                     log.info(
#                         f"      ├─ Saved gold embeddings to embeddings/img_gold.pkl")
#                 else:
#                     log.info(f"      ├─ Skipped saving empty gold embeddings")

#                 # Also save silver embeddings for future use
#                 if img_silver.size > 0:
#                     joblib.dump(img_silver, "embeddings/img_silver.pkl")
#                     log.info(
#                         f"      ├─ Saved silver embeddings to embeddings/img_silver.pkl")

#             except Exception as e:
#                 log.warning(f"      ⚠️  Failed to save embeddings: {e}")
#             img_pbar.update(1)

#         # Enhanced dimension verification with detailed logging
#         if (img_silver is not None and img_silver.size > 0) or (img_gold is not None and img_gold.size > 0):
#             log.info(f"   🔍 DIMENSION VERIFICATION:")

#             # Check silver dimensions
#             if img_silver is not None and img_silver.size > 0:
#                 log.info(f"   ├─ Silver embeddings: {img_silver.shape}")
#                 log.info(
#                     f"   ├─ Silver DataFrame: {len(img_silver_df):,} rows")

#                 # Fix dimension mismatch
#                 if img_silver.shape[0] != len(img_silver_df):
#                     log.warning(
#                         f"   ⚠️  Silver dimension mismatch: {img_silver.shape[0]} != {len(img_silver_df)}")

#                     # Option 1: If embeddings has more rows, truncate
#                     if img_silver.shape[0] > len(img_silver_df):
#                         img_silver = img_silver[:len(img_silver_df)]
#                         log.info(
#                             f"   ├─ Truncated embeddings to match DataFrame")

#                     # Option 2: If DataFrame has more rows, filter it
#                     else:
#                         # Get the first N indices where N = number of embeddings
#                         valid_indices = img_silver_df.index[:img_silver.shape[0]]
#                         img_silver_df = img_silver_df.loc[valid_indices]
#                         log.info(
#                             f"   ├─ Filtered DataFrame to match embeddings")

#             # Check gold dimensions
#             if img_gold is not None and img_gold.size > 0:
#                 log.info(f"   ├─ Gold embeddings: {img_gold.shape}")
#                 log.info(f"   └─ Gold DataFrame: {len(img_gold_df):,} rows")

#                 # Fix dimension mismatch
#                 if img_gold.shape[0] != len(img_gold_df):
#                     log.warning(
#                         f"   ⚠️  Gold dimension mismatch: {img_gold.shape[0]} != {len(img_gold_df)}")

#                     if img_gold.shape[0] > len(img_gold_df):
#                         img_gold = img_gold[:len(img_gold_df)]
#                         log.info(
#                             f"   ├─ Truncated embeddings to match DataFrame")
#                     else:
#                         valid_indices = img_gold_df.index[:img_gold.shape[0]]
#                         img_gold_df = img_gold_df.loc[valid_indices]
#                         log.info(
#                             f"   ├─ Filtered DataFrame to match embeddings")

#         # Convert to sparse matrices for memory efficiency
#         try:
#             if img_silver is not None and img_silver.size > 0:
#                 X_img_silver = csr_matrix(img_silver)
#                 log.debug(
#                     f"      ├─ Silver sparse matrix: {X_img_silver.shape}")
#             else:
#                 # Empty sparse matrix
#                 X_img_silver = None
#                 log.debug(f"      ├─ Silver: No image features available")

#             if img_gold is not None and img_gold.size > 0:
#                 X_img_gold = csr_matrix(img_gold)
#                 log.debug(f"      ├─ Gold sparse matrix: {X_img_gold.shape}")
#             else:
#                 # Empty sparse matrix
#                 X_img_gold = None
#                 log.debug(f"      ├─ Gold: No image features available")

#         except Exception as e:
#             log.error(f"   ❌ Sparse matrix conversion failed: {e}")
#             X_img_silver = None
#             X_img_gold = None

#         # Comprehensive results summary
#         log.info(f"   📊 Image Processing Results:")
#         log.info(f"   ├─ Silver images available: {len(silver_img):,}")
#         log.info(f"   ├─ Silver images downloaded: {len(silver_downloaded):,}")
#         log.info(f"   ├─ Silver valid embeddings: {img_silver.shape[0]:,}")
#         log.info(f"   ├─ Gold images available: {len(gold_img):,}")
#         log.info(f"   ├─ Gold images downloaded: {len(gold_downloaded):,}")
#         log.info(f"   ├─ Gold valid embeddings: {img_gold.shape[0]:,}")
#         log.info(
#             f"   ├─ Silver embedding size: {img_silver.nbytes // (1024**2) if img_silver.size > 0 else 0} MB")
#         log.info(
#             f"   └─ Gold embedding size: {img_gold.nbytes // (1024**2) if img_gold.size > 0 else 0} MB")

#         # Enhanced early exit logic for image-only mode
#         if mode == "image":
#             total_valid_images = img_silver.shape[0] + img_gold.shape[0]
#             min_required_images = 10  # Minimum viable images for training

#             if total_valid_images < min_required_images:
#                 log.warning(f"   ⚠️  Insufficient images for image-only mode!")
#                 log.warning(
#                     f"      ├─ Total valid images: {total_valid_images}")
#                 log.warning(
#                     f"      ├─ Minimum required: {min_required_images}")
#                 log.warning(
#                     f"      └─ Consider using mode='text' or mode='both'")

#                 stage_time = time.time() - stage_start
#                 log.info(
#                     f"   ❌ Image processing insufficient in {stage_time:.1f}s")
#                 return None, None, None, []
#             else:
#                 log.info(
#                     f"   ✅ Sufficient images for image-only mode: {total_valid_images}")

#         stage_time = time.time() - stage_start
#         log.info(f"   ✅ Image processing completed in {stage_time:.1f}s")
#         log_memory_usage("Image Processing")
#         optimize_memory_usage("Image Processing")

#     else:
#         log.info("\n⏭️  STAGE 3: SKIPPED (Image processing not requested)")

#     pipeline_progress.update(1)

#     # ------------------------------------------------------------------
#     # 4. MODEL TRAINING
#     # ------------------------------------------------------------------
#     pipeline_progress.set_description("🔬 ML Pipeline: Model Training")
#     stage_start = time.time()

#     log.info("\n🤖 STAGE 4: MODEL TRAINING")

#     training_subtasks = []
#     if mode in {"image", "both"} and img_silver is not None and img_silver.size > 0:
#         training_subtasks.append("Image Models")
#     if mode in {"text", "both"}:
#         training_subtasks.append("Text Models")
#     if mode == "both" and img_silver is not None and img_silver.size > 0:
#         training_subtasks.append("Text+Image Ensemble")
#         training_subtasks.append("Final Combined")

#     with tqdm(training_subtasks, desc="   ├─ Training Phases", position=1, leave=False,
#               bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as train_pbar:

#         # IMAGE MODELS
#         if mode in {"image", "both"}:
#             train_pbar.set_description("   ├─ Training Image Models")
#             log.info(f"   🖼️  Training image-based models...")

#             try:
#                 if mode == "both":
#                     # For "both" mode, we should use COMBINED features for image models too!
#                     log.info(
#                         "   📊 Mode='both' detected - using combined text+image features")

#                     # Find common indices between text and image data
#                     common_silver_idx = silver_txt.index.intersection(
#                         img_silver_df.index)
#                     common_gold_idx = gold.index.intersection(
#                         img_gold_df.index)

#                     if len(common_silver_idx) > 0 and len(common_gold_idx) > 0:
#                         # Create combined features for silver
#                         X_text_silver_common = vec.transform(
#                             silver_txt.loc[common_silver_idx].clean)
#                         img_silver_common = img_silver[img_silver_df.index.get_indexer(
#                             common_silver_idx)]
#                         X_combined_silver = combine_features(
#                             X_text_silver_common, img_silver_common)

#                         # Create combined features for gold
#                         X_text_gold_common = vec.transform(
#                             gold.loc[common_gold_idx].clean)
#                         img_gold_common = img_gold[img_gold_df.index.get_indexer(
#                             common_gold_idx)]
#                         X_combined_gold = combine_features(
#                             X_text_gold_common, img_gold_common)

#                         # Use aligned DataFrames
#                         silver_eval = silver_txt.loc[common_silver_idx]
#                         gold_eval = gold.loc[common_gold_idx]

#                         log.info(
#                             f"      ├─ Combined features for image models: {X_combined_silver.shape}")
#                         log.info(
#                             f"      ├─ Samples: {len(common_silver_idx)} silver, {len(common_gold_idx)} gold")

#                         res_img = run_mode_A(
#                             X_combined_silver,  # Combined features!
#                             gold_eval.clean,
#                             X_combined_gold,    # Combined features!
#                             silver_eval,
#                             gold_eval,
#                             domain="image",  # Still marked as "image" domain for model selection
#                             apply_smote_flag=False
#                         )
#                     else:
#                         log.warning(
#                             "   ⚠️  No common indices between text and image data")
#                         res_img = []
#                 else:
#                     # For pure "image" mode, use only image features
#                     log.info("   📊 Mode='image' - using image-only features")
#                     res_img = run_mode_A(
#                         X_img_silver,
#                         img_gold_df.clean,
#                         X_img_gold,
#                         img_silver_df,
#                         img_gold_df,
#                         domain="image",
#                         apply_smote_flag=False
#                     )

#                 results.extend(res_img)
#                 log.info(f"      ✅ Image models: {len(res_img)} results")

#             except Exception as e:
#                 log.error(f"      ❌ Image model training failed: {str(e)}")
#                 import traceback
#                 log.error(f"Full traceback:\n{traceback.format_exc()}")

#             optimize_memory_usage("Image Models")
#             train_pbar.update(1)
#         # TEXT MODELS
#         if mode in {"text", "both"}:
#             train_pbar.set_description("   ├─ Training Text Models")
#             log.info(f"   🔤 Training text-based models...")

#             res_text = run_mode_A(
#                 X_text_silver, gold.clean, X_text_gold,
#                 silver_txt, gold,
#                 domain="text", apply_smote_flag=True
#             )

#             results.extend(res_text)
#             log.info(f"      ✅ Text models: {len(res_text)} results")
#             optimize_memory_usage("Text Models")
#             train_pbar.update(1)

#         # TEXT+IMAGE ENSEMBLE
#         if mode == "both" and len(res_text) > 0 and len(res_img) > 0:
#             train_pbar.set_description("   ├─ Text+Image Ensemble")
#             log.info(f"   🤝 Creating text+image ensemble...")

#             ensemble_results = []
#             alpha_values = [0.25, 0.5, 0.75]

#             for task in ("keto", "vegan"):
#                 try:
#                     # Check if we have models for both domains
#                     text_models = [r for r in res_text if r["task"] == task]
#                     image_models = [r for r in res_img if r["task"] == task]

#                     if not text_models:
#                         log.warning(
#                             f"      ⚠️  No text models available for {task}")
#                         continue

#                     if not image_models:
#                         log.warning(
#                             f"      ⚠️  No image models available for {task}")
#                         # Still create ensemble with text-only
#                         best_text_result = max(
#                             text_models, key=lambda x: x['F1'])
#                         ensemble_results.append(best_text_result)
#                         continue

#                     log.info(
#                         f"      ├─ {task}: Testing alpha values {alpha_values}")

#                     best_ensemble_result = None
#                     best_f1 = -1
#                     best_alpha = None

#                     for alpha in alpha_values:
#                         try:
#                             result = best_two_domains(
#                                 task=task,
#                                 text_results=res_text,
#                                 image_results=res_img,
#                                 gold_df=gold,  # Use full gold df (100 samples)
#                                 alpha=alpha
#                             )

#                             if result and result['F1'] > best_f1:
#                                 best_f1 = result['F1']
#                                 best_alpha = alpha
#                                 best_ensemble_result = result

#                             log.info(
#                                 f"         ├─ alpha={alpha}: F1={result['F1']:.3f}")

#                         except Exception as e:
#                             log.error(
#                                 f"         ❌ alpha={alpha} failed: {str(e)[:50]}...")
#                             continue

#                     if best_ensemble_result:
#                         best_ensemble_result['model'] = f"BestTwo_alpha{best_alpha}"
#                         ensemble_results.append(best_ensemble_result)
#                         log.info(
#                             f"      ✅ {task} best ensemble: alpha={best_alpha}, F1={best_f1:.3f}")
#                     else:
#                         # Fallback to best text model
#                         log.warning(
#                             f"      ⚠️  Ensemble failed for {task}, using best text model")
#                         best_text = max(text_models, key=lambda x: x['F1'])
#                         ensemble_results.append(best_text)

#                 except Exception as e:
#                     log.error(
#                         f"      ❌ {task} ensemble creation failed: {str(e)[:50]}...")
#                     continue

#             if ensemble_results:
#                 table("Text+Image Ensembles", ensemble_results)
#                 results.extend(ensemble_results)
#                 log.info(
#                     f"      ✅ Created {len(ensemble_results)} ensemble configurations")
#             else:
#                 log.warning(f"      ⚠️  No successful ensembles created")

#             train_pbar.update(1)

#         # FINAL COMBINED MODEL TRAINING
#         if mode == "both":
#             # Check if we have valid image data
#             has_silver_images = (img_silver is not None) and (
#                 hasattr(img_silver, 'size')) and (img_silver.size > 0)
#             has_gold_images = (img_gold is not None) and (
#                 hasattr(img_gold, 'size')) and (img_gold.size > 0)

#             if has_silver_images and has_gold_images:
#                 train_pbar.set_description("   ├─ Final Combined Models")
#                 log.info(f"   🔄 Training final combined models...")

#             # Align features and data. Find the intersection of indices between text and image data
#             common_silver_idx = silver_txt.index.intersection(
#                 img_silver_df.index)
#             common_gold_idx = gold.index.intersection(img_gold_df.index)

#             if len(common_silver_idx) > 0 and len(common_gold_idx) > 0:
#                 # Align silver features
#                 X_text_silver_algn = vec.transform(
#                     silver_txt.loc[common_silver_idx].clean)
#                 # Also need to align the image embeddings to the common indices
#                 img_silver_aligned = img_silver[img_silver_df.index.get_indexer(
#                     common_silver_idx)]
#                 X_silver = combine_features(
#                     X_text_silver_algn, img_silver_aligned)

#                 # Align gold features
#                 X_text_gold_algn = vec.transform(
#                     gold.loc[common_gold_idx].clean)
#                 # Also align gold image embeddings
#                 img_gold_aligned = img_gold[img_gold_df.index.get_indexer(
#                     common_gold_idx)]
#                 X_gold = combine_features(X_text_gold_algn, img_gold_aligned)

#                 silver_eval = silver_txt.loc[common_silver_idx]
#                 gold_eval = gold.loc[common_gold_idx]

#                 log.info(
#                     f"      ├─ Combined silver features: {X_silver.shape}")
#                 log.info(f"      ├─ Combined gold features: {X_gold.shape}")
#                 log.info(f"      ├─ Silver samples: {len(silver_eval):,}")
#                 log.info(f"      └─ Gold samples: {len(gold_eval):,}")

#                 # Run combined training
#                 res_combined = run_mode_A(
#                     X_silver, gold_eval.clean, X_gold,
#                     silver_eval, gold_eval,
#                     domain="both", apply_smote_flag=True
#                 )
#                 results.extend(res_combined)
#                 log.info(
#                     f"      ✅ Combined models: {len(res_combined)} results")
#                 optimize_memory_usage()

#             else:
#                 log.warning(
#                     f"      ⚠️  No common indices for combined features, skipping")

#             train_pbar.update(1)

#         # Setup feature matrices for ensemble creation
#         if mode == "both" and img_silver is not None and img_silver.size > 0:
#             X_silver, X_gold = X_silver, X_gold
#             silver_eval = silver_eval
#         elif mode == "text":
#             X_silver, X_gold = X_text_silver, X_text_gold
#             silver_eval = silver_txt
#         elif mode == "image" and img_silver is not None and img_silver.size > 0:
#             X_silver, X_gold = csr_matrix(img_silver), csr_matrix(img_gold)
#             silver_eval = img_silver_df
#         else:
#             # Fallback to text
#             log.warning(
#                 f"   ⚠️  No valid images for image mode, falling back to text")
#             X_silver, X_gold = X_text_silver, X_text_gold
#             silver_eval = silver_txt

#         # Final training if no results yet
#         if not results:
#             log.info(f"   🎯 Running fallback text-only training...")
#             res_final = run_mode_A(X_text_silver, gold.clean, X_text_gold,
#                                    silver_txt, gold, domain="text", apply_smote_flag=True)
#             results.extend(res_final)
#             log.info(f"      ✅ Final models: {len(res_final)} results")

#     stage_time = time.time() - stage_start
#     log.info(f"   ✅ Model training completed in {stage_time:.1f}s")
#     log.info(f"   📊 Total models trained: {len(results)}")
#     log_memory_usage("Model Training")
#     pipeline_progress.update(1)

#     # ------------------------------------------------------------------
#     # 5. ENSEMBLE OPTIMIZATION
#     # ------------------------------------------------------------------
#     pipeline_progress.set_description("🔬 ML Pipeline: Ensemble Creation")
#     stage_start = time.time()

#     log.info("\n🎭 STAGE 5: ENSEMBLE OPTIMIZATION")

#     if len(results) > 0:
#         ensemble_tasks = ["keto", "vegan"]
#         with tqdm(ensemble_tasks, desc="   ├─ Ensemble Tasks", position=1, leave=False,
#                   bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}<{remaining}]") as ens_pbar:

#             ensemble_results = []
#             for task in ens_pbar:
#                 ens_pbar.set_description(f"   ├─ Optimizing {task} ensemble")

#                 log.info(f"   🎯 Optimizing {task} ensemble...")

#                 # Count available models
#                 task_models = [r for r in results if r["task"]
#                                == task and r["model"] != "Rule"]
#                 log.info(f"      ├─ Available models: {len(task_models)}")

#                 if len(task_models) > 1:
#                     # Use appropriate features
#                     if mode == "both" and img_silver is not None and img_silver.size > 0:
#                         ens_X_silver = X_silver
#                         ens_X_gold = X_gold
#                         ens_silver_eval = silver_eval
#                     elif mode == "image" and img_silver is not None and img_silver.size > 0:
#                         ens_X_silver = csr_matrix(img_silver)
#                         ens_X_gold = csr_matrix(img_gold)
#                         ens_silver_eval = img_silver_df
#                     else:
#                         ens_X_silver = X_text_silver
#                         ens_X_gold = X_text_gold
#                         ens_silver_eval = silver_txt

#                     best_ens = best_ensemble(task, results, ens_X_silver, gold.clean,
#                                              ens_X_gold, ens_silver_eval, gold)
#                     if best_ens:
#                         ensemble_results.append(best_ens)
#                         log.info(
#                             f"      ✅ {task} ensemble: {best_ens['model']} (F1={best_ens['F1']:.3f})")
#                     else:
#                         log.warning(
#                             f"      ⚠️  {task} ensemble optimization failed")
#                 else:
#                     log.info(
#                         f"      ⏭️  {task}: Only {len(task_models)} model(s) available, skipping ensemble")

#             results.extend(ensemble_results)
#             log.info(
#                 f"   📊 Ensemble results: {len(ensemble_results)} optimized ensembles")

#         if mode == "both" and 'res_text' in locals() and 'res_img' in locals():
#             log.info(f"\n   🔀 Cross-domain ensemble optimization...")

#             cross_domain_results = []
#             for task in ensemble_tasks:
#                 try:
#                     # Check we have both text and image results
#                     text_task_models = [
#                         r for r in res_text if r["task"] == task]
#                     image_task_models = [
#                         r for r in res_img if r["task"] == task]

#                     if text_task_models and image_task_models:
#                         log.info(
#                             f"      ├─ Optimizing {task} cross-domain ensemble...")

#                         # Use best_ensemble with image_res for smart blending
#                         cross_result = best_ensemble(
#                             task=task,
#                             res=res_text,  # Text results as primary
#                             X_vec=X_text_silver,
#                             clean=gold.clean,
#                             X_gold=X_text_gold,
#                             silver=silver_txt,
#                             gold=gold,
#                             image_res=res_img,  # This enables smart text+image blending
#                             alphas=(0.25, 0.5, 0.75)  # Alpha values to test
#                         )

#                         if cross_result:
#                             cross_domain_results.append(cross_result)
#                             log.info(f"      ✅ {task} cross-domain: "
#                                      f"{cross_result['model']} (F1={cross_result['F1']:.3f})")
#                     else:
#                         log.info(
#                             f"      ⏭️  {task}: Missing text or image models for cross-domain")

#                 except Exception as e:
#                     log.error(
#                         f"      ❌ {task} cross-domain failed: {str(e)[:50]}...")

#             if cross_domain_results:
#                 results.extend(cross_domain_results)
#                 log.info(
#                     f"   📊 Cross-domain results: {len(cross_domain_results)} optimized ensembles")

#     else:
#         log.warning(f"   ⚠️  No models available for ensemble optimization")

#     stage_time = time.time() - stage_start
#     log.info(f"   ✅ Ensemble optimization completed in {stage_time:.1f}s")
#     log_memory_usage("Ensemble Creation")
#     pipeline_progress.update(1)

#     # ------------------------------------------------------------------
#     # 6. EVALUATION AND EXPORT
#     # ------------------------------------------------------------------
#     pipeline_progress.set_description("🔬 ML Pipeline: Evaluation")
#     stage_start = time.time()

#     log.info("\n📊 STAGE 6: EVALUATION AND EXPORT")

#     with tqdm(total=3, desc="   ├─ Export Process", position=1, leave=False,
#               bar_format="   ├─ {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]") as export_pbar:

#         export_pbar.set_description("   ├─ Generating plots")
#         if len(results) > 0:
#             export_eval_plots(results, gold)
#             log.info(f"      ✅ Generated evaluation plots and confusion matrices")
#         else:
#             log.warning(f"      ⚠️  No results to plot")
#         export_pbar.update(1)

#         export_pbar.set_description("   ├─ Saving results")
#         # Save results summary
#         results_summary = []
#         for r in results:
#             summary = {
#                 'task': r['task'],
#                 'model': r['model'],
#                 'f1': r['F1'],
#                 'accuracy': r['ACC'],
#                 'precision': r['PREC'],
#                 'recall': r['REC'],
#                 'roc_auc': r['ROC'],
#                 'pr_auc': r['PR']
#             }
#             results_summary.append(summary)

#         if results_summary:
#             pd.DataFrame(results_summary).to_csv(
#                 "pipeline_results_summary.csv", index=False)
#             log.info(
#                 f"      ✅ Saved results summary with {len(results_summary)} entries")
#         else:
#             log.warning(f"      ⚠️  No results to save")
#         export_pbar.update(1)

#         export_pbar.set_description("   ├─ Cleanup")
#         gc.collect()
#         export_pbar.update(1)

#     stage_time = time.time() - stage_start
#     log.info(f"   ✅ Evaluation completed in {stage_time:.1f}s")
#     log_memory_usage("Final")
#     pipeline_progress.update(1)

# # ------------------------------------------------------------------
#     # PIPELINE COMPLETION SUMMARY
# # ------------------------------------------------------------------
#     total_time = time.time() - pipeline_start

#     log.info(f"\n🏁 PIPELINE COMPLETE")
#     log.info(
#         f"   ├─ Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
#     log.info(f"   ├─ Mode: {mode}")
#     log.info(f"   ├─ Sample fraction: {sample_frac or 'Full dataset'}")
#     log.info(f"   ├─ Total results: {len(results)}")
#     log.info(f"   └─ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     # Performance summary
#     if results:
#         log.info(f"\n   🏆 FINAL PERFORMANCE SUMMARY:")
#         for task in ["keto", "vegan"]:
#             task_results = [r for r in results if r["task"] == task]
#             if task_results:
#                 best_result = max(task_results, key=lambda x: x['F1'])
#                 log.info(f"   ├─ {task.upper()}: Best F1={best_result['F1']:.3f} "
#                          f"({best_result['model']}) | ACC={best_result['ACC']:.3f}")

#         # Organize results by domain for better analysis
#         results_by_domain = {}
#         for res in results:
#             domain = res.get('domain', 'unknown')
#             if domain not in results_by_domain:
#                 results_by_domain[domain] = []
#             results_by_domain[domain].append(res)

#         # Export aggregated results across all domains
#         aggregate_results_across_domains(results_by_domain)

#         # Export ensemble-specific metrics if any ensembles were created
#         ensemble_results = [r for r in results if
#                             'ensemble' in r.get('domain', '') or
#                             'Ens' in r.get('model', '') or
#                             'BestTwo' in r.get('model', '') or
#                             'SmartEns' in r.get('model', '')]

#         if ensemble_results:
#             export_ensemble_metrics(ensemble_results)
#             log.info(f"\n   🎭 ENSEMBLE SUMMARY:")
#             log.info(f"   ├─ Total ensembles created: {len(ensemble_results)}")

#             # Best ensemble per task
#             for task in ["keto", "vegan"]:
#                 task_ensembles = [
#                     r for r in ensemble_results if r["task"] == task]
#                 if task_ensembles:
#                     best_ensemble = max(task_ensembles, key=lambda x: x['F1'])
#                     log.info(
#                         f"   ├─ {task.upper()} best ensemble: {best_ensemble['model']} (F1={best_ensemble['F1']:.3f})")

#         # Create comprehensive pipeline summary
#         pipeline_summary = {
#             'mode': mode,
#             'sample_frac': sample_frac,
#             'total_time': total_time,
#             'total_time_minutes': total_time / 60,
#             'total_models': len(results),
#             'total_ensembles': len(ensemble_results),
#             'domains_trained': list(results_by_domain.keys()),
#             'models_per_domain': {domain: len(models) for domain, models in results_by_domain.items()},
#             'best_keto_f1': max([r['F1'] for r in results if r['task'] == 'keto'], default=0),
#             'best_vegan_f1': max([r['F1'] for r in results if r['task'] == 'vegan'], default=0),
#             'best_keto_model': max([r for r in results if r['task'] == 'keto'], key=lambda x: x['F1'], default={'model': 'None'})['model'],
#             'best_vegan_model': max([r for r in results if r['task'] == 'vegan'], key=lambda x: x['F1'], default={'model': 'None'})['model'],
#             'timestamp': datetime.now().isoformat(),
#             'pipeline_stages_completed': [
#                 'data_loading', 'text_processing',
#                 'image_processing' if mode in ['image', 'both'] else None,
#                 'model_training', 'ensemble_creation', 'evaluation'
#             ]
#         }

#         # Save detailed pipeline summary
#         summary_dir = CFG.artifacts_dir / "metrics"
#         summary_dir.mkdir(parents=True, exist_ok=True)
#         summary_path = summary_dir / \
#             f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

#         with open(summary_path, 'w') as f:
#             json.dump(pipeline_summary, f, indent=2)
#         log.info(f"\n   📝 Pipeline summary saved to {summary_path}")

#     else:
#         log.warning(f"\n   ⚠️  NO RESULTS GENERATED")
#         log.warning(
#             f"   └─ Consider checking data availability or adjusting parameters")

#     # Resource usage summary
#     final_memory = psutil.virtual_memory()
#     log.info(f"\n   💾 RESOURCE USAGE:")
#     log.info(f"   ├─ Peak memory: {final_memory.percent:.1f}%")
#     log.info(f"   ├─ Final memory: {final_memory.used // (1024**2)} MB")
#     log.info(f"   └─ Efficiency: {len(results)/total_time:.2f} models/second")

#     # Enhanced pipeline metadata with more details
#     pipeline_metadata = {
#         'mode': mode,
#         'force': force,
#         'sample_frac': sample_frac,
#         'total_time': total_time,
#         'total_time_minutes': total_time / 60,
#         'total_results': len(results),
#         'start_time': pipeline_start,
#         'end_time': time.time(),
#         'memory_peak_percent': final_memory.percent,
#         'memory_used_mb': final_memory.used // (1024**2),
#         'system_info': {
#             'cpu_count': psutil.cpu_count(),
#             'total_memory_gb': psutil.virtual_memory().total // (1024**3)
#         },
#         'data_stats': {
#             'silver_text_size': len(silver_txt),
#             'silver_image_size': len(silver_img),
#             'gold_size': len(gold),
#             'silver_images_downloaded': len(silver_downloaded) if 'silver_downloaded' in locals() else 0,
#             'gold_images_downloaded': len(gold_downloaded) if 'gold_downloaded' in locals() else 0,
#             'image_embeddings_used': {
#                 'silver': img_silver.shape[0] if 'img_silver' in locals() and img_silver is not None else 0,
#                 'gold': img_gold.shape[0] if 'img_gold' in locals() and img_gold is not None else 0
#             }
#         },
#         'model_summary': {
#             'total_models': len(results),
#             'text_models': len([r for r in results if r.get('domain') == 'text']),
#             'image_models': len([r for r in results if r.get('domain') == 'image']),
#             'combined_models': len([r for r in results if r.get('domain') == 'both']),
#             'ensemble_models': len([r for r in results if 'ensemble' in r.get('domain', '') or 'Ens' in r.get('model', '')])
#         },
#         'performance_summary': {
#             'best_overall_f1': max([r['F1'] for r in results], default=0),
#             'best_overall_model': max(results, key=lambda x: x['F1'], default={'model': 'None'})['model'] if results else 'None',
#             'average_f1': np.mean([r['F1'] for r in results]) if results else 0,
#             'average_training_time': np.mean([r.get('training_time', 0) for r in results]) if results else 0
#         }
#     }

#     # Save to both JSON and CSV for easy access
#     metadata_path = CFG.artifacts_dir / "metrics" / \
#         f"pipeline_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     with open(metadata_path, "w") as f:
#         json.dump(pipeline_metadata, f, indent=2)

#     # Also save a simplified version to the root for backward compatibility
#     with open("pipeline_metadata.json", "w") as f:
#         json.dump(pipeline_metadata, f, indent=2)

#     log.info(f"   💾 Saved pipeline metadata to {metadata_path}")

#     # Create a master results DataFrame for easy analysis
#     if results:
#         master_results = []
#         for res in results:
#             master_results.append({
#                 'timestamp': datetime.now().isoformat(),
#                 'pipeline_mode': mode,
#                 'task': res['task'],
#                 'model': res['model'],
#                 'domain': res.get('domain', 'unknown'),
#                 'f1_score': res['F1'],
#                 'accuracy': res['ACC'],
#                 'precision': res['PREC'],
#                 'recall': res['REC'],
#                 'roc_auc': res['ROC'],
#                 'pr_auc': res['PR'],
#                 'training_time': res.get('training_time', 0)
#             })

#         master_df = pd.DataFrame(master_results)
#         master_path = CFG.artifacts_dir / "metrics" / \
#             f"master_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         master_df.to_csv(master_path, index=False)
#         log.info(f"   📊 Saved master results table to {master_path}")

#         # Print final summary statistics
#         log.info(f"\n   📈 FINAL STATISTICS:")
#         log.info(f"   ├─ Models trained per task:")
#         for task in ['keto', 'vegan']:
#             task_count = len([r for r in results if r['task'] == task])
#             log.info(f"   │  ├─ {task}: {task_count} models")
#         log.info(
#             f"   ├─ Average F1 score: {pipeline_metadata['performance_summary']['average_f1']:.3f}")
#         log.info(
#             f"   ├─ Best overall F1: {pipeline_metadata['performance_summary']['best_overall_f1']:.3f}")
#         log.info(
#             f"   └─ Total training efficiency: {len(results)/total_time:.2f} models/second")

#     return vec, silver_txt, gold, results

