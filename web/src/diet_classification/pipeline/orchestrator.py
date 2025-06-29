#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline orchestrator for diet classification.

This module coordinates all components of the ML pipeline including:
- Data loading and preparation
- Feature extraction (text and images)
- Model training and evaluation
- Ensemble creation
- Result export and visualization
"""

import time
import json
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ..core import log, get_pipeline_state
from ..config import CFG
from ..data import get_datasets, show_balance
from ..features.text import extract_text_features
from ..features.images import download_images, build_image_embeddings
from ..features.combiners import combine_features, filter_photo_rows
from ..models.training import run_mode_A
from ..models.io import save_models_optimized
from ..ensembles import (
    best_two_domains,
    top_n_ensemble,
    best_ensemble,
    create_voting_ensemble
)
from ..evaluation import (
    export_results_to_csv,
    aggregate_results_across_domains,
    export_ensemble_metrics,
    export_eval_plots,
    table
)
from ..utils.memory import optimize_memory_usage, get_available_memory
from ..utils.validation import preflight_checks
from ..pipeline.checkpoints import save_pipeline_state, load_pipeline_state


def run_full_pipeline(
    mode: str = "both",
    force: bool = False,
    sample_frac: Optional[float] = None,
    checkpoint_resume: bool = True
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Execute the complete machine learning pipeline for diet classification.
    
    This is the main orchestration function that coordinates all pipeline stages:
    1. Pre-flight checks and initialization
    2. Data loading and preparation
    3. Text feature extraction (TF-IDF)
    4. Image feature extraction (ResNet-50)
    5. Model training on different feature types
    6. Ensemble creation and optimization
    7. Evaluation and result export
    
    Args:
        mode: Feature mode - 'text', 'image', or 'both'
        force: Force recomputation of cached features
        sample_frac: Fraction of data to sample (for testing)
        checkpoint_resume: Whether to resume from checkpoints
        
    Returns:
        Tuple of (vectorizer, silver_df, gold_df, results)
    """
    # Initialize tracking
    pipeline_start = time.time()
    pipeline_state = get_pipeline_state()
    
    # Log initialization
    log.info("="*80)
    log.info("🚀 STARTING FULL MACHINE LEARNING PIPELINE")
    log.info("="*80)
    log.info(f"   Configuration:")
    log.info(f"   ├─ Mode: {mode}")
    log.info(f"   ├─ Force recompute: {force}")
    log.info(f"   ├─ Sample fraction: {sample_frac or 'Full dataset'}")
    log.info(f"   ├─ Checkpoint resume: {checkpoint_resume}")
    log.info(f"   ├─ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"   ├─ Available CPU cores: {psutil.cpu_count()}")
    log.info(f"   └─ Available memory: {get_available_memory():.1f} GB")
    
    # Run pre-flight checks
    log.info("\n" + "="*60)
    log.info("STAGE 0: PRE-FLIGHT CHECKS")
    log.info("="*60)
    
    if not preflight_checks():
        log.error("❌ Pre-flight checks failed. Aborting pipeline.")
        raise RuntimeError("Pre-flight checks failed")
    
    # Check for saved pipeline state
    if checkpoint_resume:
        saved_stage, saved_data = load_pipeline_state()
        if saved_stage:
            log.info(f"\n📂 Found saved pipeline state from stage: {saved_stage}")
            response = input("Resume from saved state? [Y/n]: ").strip().lower()
            if response != 'n':
                log.info("Resuming from checkpoint...")
                # Restore state will be handled by individual stages
    
    # Pipeline stages with progress tracking
    stages = [
        "Data Loading",
        "Text Processing", 
        "Image Processing",
        "Model Training",
        "Ensemble Creation",
        "Evaluation & Export"
    ]
    
    # Main progress bar
    pipeline_progress = tqdm(
        stages, 
        desc="🔬 Pipeline Progress",
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    
    # Track results
    results = []
    results_by_domain = {"text": [], "image": [], "both": [], "ensemble": []}
    
    # =========================================================================
    # STAGE 1: DATA LOADING AND PREPARATION
    # =========================================================================
    pipeline_progress.set_description("🔬 Pipeline: Data Loading")
    stage_start = time.time()
    
    log.info("\n" + "="*60)
    log.info("STAGE 1: DATA LOADING AND PREPARATION")
    log.info("="*60)
    
    # Load datasets
    silver_all, gold, recipes, carb_df = get_datasets(sample_frac)
    
    # Prepare data subsets
    silver_all["uid"] = silver_all.index
    gold["uid"] = gold.index
    
    # Text data (all samples)
    silver_txt = silver_all.copy()
    
    # Image data (filtered by photo availability)
    silver_img = filter_photo_rows(silver_all)
    gold_img = filter_photo_rows(gold)
    
    # Log statistics
    log.info(f"\n📊 Dataset Statistics:")
    log.info(f"   ├─ Silver (All): {len(silver_all):,} recipes")
    log.info(f"   ├─ Silver (Text): {len(silver_txt):,} recipes")
    log.info(f"   ├─ Silver (Images): {len(silver_img):,} recipes")
    log.info(f"   ├─ Gold (All): {len(gold):,} recipes")
    log.info(f"   └─ Gold (Images): {len(gold_img):,} recipes")
    
    # Show class balance
    show_balance(silver_txt, "Silver")
    show_balance(gold, "Gold")
    
    stage_time = time.time() - stage_start
    log.info(f"\n✅ Data loading completed in {stage_time:.1f}s")
    
    # Save checkpoint
    save_pipeline_state("data_loading", {
        "silver_all": silver_all,
        "gold": gold,
        "silver_txt": silver_txt,
        "silver_img": silver_img,
        "gold_img": gold_img
    })
    
    optimize_memory_usage("After data loading")
    pipeline_progress.update(1)
    
    # =========================================================================
    # STAGE 2: TEXT FEATURE PROCESSING
    # =========================================================================
    pipeline_progress.set_description("🔬 Pipeline: Text Processing")
    stage_start = time.time()
    
    log.info("\n" + "="*60)
    log.info("STAGE 2: TEXT FEATURE EXTRACTION")
    log.info("="*60)
    
    # Extract text features with USDA integration
    X_text_silver, X_text_gold, vectorizer, silver_extended = extract_text_features(
        silver_txt, 
        gold,
        include_usda=True,
        sample_frac=None  # Already sampled
    )
    
    # Update silver_txt with extended data
    silver_txt = silver_extended
    
    # Store in pipeline state
    pipeline_state.vectorizer = vectorizer
    
    stage_time = time.time() - stage_start
    log.info(f"\n✅ Text processing completed in {stage_time:.1f}s")
    
    # Save checkpoint
    save_pipeline_state("text_processing", {
        "vectorizer": vectorizer,
        "X_text_silver_shape": X_text_silver.shape,
        "X_text_gold_shape": X_text_gold.shape,
        "vocab_size": len(vectorizer.vocabulary_)
    })
    
    optimize_memory_usage("After text processing")
    pipeline_progress.update(1)
    
    # =========================================================================
    # STAGE 3: IMAGE FEATURE PROCESSING (if applicable)
    # =========================================================================
    X_img_silver = None
    X_img_gold = None
    img_silver_df = pd.DataFrame()
    img_gold_df = pd.DataFrame()
    
    if mode in ["image", "both"]:
        pipeline_progress.set_description("🔬 Pipeline: Image Processing")
        stage_start = time.time()
        
        log.info("\n" + "="*60)
        log.info("STAGE 3: IMAGE FEATURE EXTRACTION")
        log.info("="*60)
        
        # Download images
        log.info("\n📥 Downloading images...")
        silver_downloaded = download_images(
            silver_img, 
            CFG.image_dir / "silver",
            force=force
        )
        gold_downloaded = download_images(
            gold_img,
            CFG.image_dir / "gold", 
            force=force
        )
        
        # Filter to downloaded images
        if silver_downloaded:
            img_silver_df = silver_img.loc[silver_downloaded].copy()
        if gold_downloaded:
            img_gold_df = gold_img.loc[gold_downloaded].copy()
        
        log.info(f"\n📊 Downloaded images:")
        log.info(f"   ├─ Silver: {len(img_silver_df):,}/{len(silver_img):,}")
        log.info(f"   └─ Gold: {len(img_gold_df):,}/{len(gold_img):,}")
        
        # Extract embeddings
        if len(img_silver_df) > 0 and len(img_gold_df) > 0:
            log.info("\n🧠 Extracting image embeddings...")
            
            # Silver embeddings
            img_silver, silver_valid_idx = build_image_embeddings(
                img_silver_df, "silver", force
            )
            
            # Gold embeddings
            img_gold, gold_valid_idx = build_image_embeddings(
                img_gold_df, "gold", force
            )
            
            # Update DataFrames to match embeddings
            img_silver_df = img_silver_df.loc[silver_valid_idx].copy()
            img_gold_df = img_gold_df.loc[gold_valid_idx].copy()
            
            # Convert to sparse matrices
            X_img_silver = csr_matrix(img_silver)
            X_img_gold = csr_matrix(img_gold)
            
            log.info(f"\n📊 Image embeddings:")
            log.info(f"   ├─ Silver: {X_img_silver.shape}")
            log.info(f"   └─ Gold: {X_img_gold.shape}")
        else:
            log.warning("⚠️  Insufficient images for processing")
            if mode == "image":
                log.error("❌ Cannot proceed with image-only mode")
                return None, None, None, []
        
        stage_time = time.time() - stage_start
        log.info(f"\n✅ Image processing completed in {stage_time:.1f}s")
        
        # Save checkpoint
        save_pipeline_state("image_processing", {
            "img_silver_shape": X_img_silver.shape if X_img_silver is not None else None,
            "img_gold_shape": X_img_gold.shape if X_img_gold is not None else None,
            "silver_downloaded": len(silver_downloaded) if silver_downloaded else 0,
            "gold_downloaded": len(gold_downloaded) if gold_downloaded else 0
        })
        
        optimize_memory_usage("After image processing")
    else:
        log.info("\n⏭️  Skipping image processing (not requested)")
    
    pipeline_progress.update(1)
    
    # =========================================================================
    # STAGE 4: MODEL TRAINING
    # =========================================================================
    pipeline_progress.set_description("🔬 Pipeline: Model Training")
    stage_start = time.time()
    
    log.info("\n" + "="*60)
    log.info("STAGE 4: MODEL TRAINING")
    log.info("="*60)
    
    # Determine training configurations
    training_configs = []
    
    if mode in ["text", "both"]:
        training_configs.append({
            "name": "Text Models",
            "X_train": X_text_silver,
            "X_test": X_text_gold,
            "train_df": silver_txt,
            "test_df": gold,
            "domain": "text",
            "apply_smote": True
        })
    
    if mode in ["image", "both"] and X_img_silver is not None:
        training_configs.append({
            "name": "Image Models",
            "X_train": X_img_silver,
            "X_test": X_img_gold,
            "train_df": img_silver_df,
            "test_df": img_gold_df,
            "domain": "image",
            "apply_smote": False
        })
    
    if mode == "both" and X_img_silver is not None:
        # Combined features for rows with both text and images
        common_silver_idx = silver_txt.index.intersection(img_silver_df.index)
        common_gold_idx = gold.index.intersection(img_gold_df.index)
        
        if len(common_silver_idx) > 0 and len(common_gold_idx) > 0:
            # Align features
            X_text_silver_common = vectorizer.transform(
                silver_txt.loc[common_silver_idx]["clean"]
            )
            X_text_gold_common = vectorizer.transform(
                gold.loc[common_gold_idx]["clean"]
            )
            
            img_silver_aligned = img_silver[
                img_silver_df.index.get_indexer(common_silver_idx)
            ]
            img_gold_aligned = img_gold[
                img_gold_df.index.get_indexer(common_gold_idx)
            ]
            
            # Combine features
            X_combined_silver = combine_features(X_text_silver_common, img_silver_aligned)
            X_combined_gold = combine_features(X_text_gold_common, img_gold_aligned)
            
            training_configs.append({
                "name": "Combined Models",
                "X_train": X_combined_silver,
                "X_test": X_combined_gold,
                "train_df": silver_txt.loc[common_silver_idx],
                "test_df": gold.loc[common_gold_idx],
                "domain": "both",
                "apply_smote": True
            })
    
    # Train models for each configuration
    with tqdm(training_configs, desc="   Training Phases", position=1, leave=False) as train_pbar:
        for config in train_pbar:
            train_pbar.set_description(f"   Training {config['name']}")
            
            log.info(f"\n🤖 Training {config['name']}...")
            log.info(f"   ├─ Domain: {config['domain']}")
            log.info(f"   ├─ Train shape: {config['X_train'].shape}")
            log.info(f"   └─ Test shape: {config['X_test'].shape}")
            
            # Run training
            domain_results = run_mode_A(
                X_silver=config["X_train"],
                gold_clean=config["test_df"]["clean"],
                X_gold=config["X_test"],
                silver_df=config["train_df"],
                gold_df=config["test_df"],
                domain=config["domain"],
                apply_smote_flag=config["apply_smote"]
            )
            
            # Store results
            results.extend(domain_results)
            results_by_domain[config["domain"]].extend(domain_results)
            
            log.info(f"   ✅ {config['name']}: {len(domain_results)} models trained")
            
            optimize_memory_usage(f"After {config['name']}")
    
    stage_time = time.time() - stage_start
    log.info(f"\n✅ Model training completed in {stage_time:.1f}s")
    log.info(f"   Total models trained: {len(results)}")
    
    # Save checkpoint
    save_pipeline_state("model_training", {
        "total_models": len(results),
        "models_by_domain": {k: len(v) for k, v in results_by_domain.items()}
    })
    
    pipeline_progress.update(1)
    
    # =========================================================================
    # STAGE 5: ENSEMBLE CREATION
    # =========================================================================
    pipeline_progress.set_description("🔬 Pipeline: Ensemble Creation")
    stage_start = time.time()
    
    log.info("\n" + "="*60)
    log.info("STAGE 5: ENSEMBLE CREATION AND OPTIMIZATION")
    log.info("="*60)
    
    ensemble_results = []
    
    # Create ensembles for each task
    for task in ["keto", "vegan"]:
        log.info(f"\n🎭 Creating ensembles for {task}...")
        
        # 1. Best N ensemble within each domain
        for domain, domain_results in results_by_domain.items():
            if domain_results:
                task_models = [r for r in domain_results if r["task"] == task]
                if len(task_models) > 1:
                    log.info(f"\n   📊 Optimizing {domain} ensemble for {task}...")
                    
                    # Get appropriate features for ensemble
                    if domain == "text":
                        X_train, X_test = X_text_silver, X_text_gold
                        train_df, test_df = silver_txt, gold
                    elif domain == "image" and X_img_silver is not None:
                        X_train, X_test = X_img_silver, X_img_gold
                        train_df, test_df = img_silver_df, img_gold_df
                    elif domain == "both" and 'X_combined_silver' in locals():
                        X_train, X_test = X_combined_silver, X_combined_gold
                        train_df, test_df = silver_txt.loc[common_silver_idx], gold.loc[common_gold_idx]
                    else:
                        continue
                    
                    # Find best ensemble size
                    best_ens = best_ensemble(
                        task=task,
                        results=task_models,
                        X_train=X_train,
                        clean=test_df["clean"],
                        X_test=X_test,
                        silver=train_df,
                        gold=test_df
                    )
                    
                    if best_ens:
                        best_ens["domain"] = "ensemble"
                        ensemble_results.append(best_ens)
                        log.info(f"      ✅ {domain} ensemble: {best_ens['model']} (F1={best_ens['F1']:.3f})")
        
        # 2. Cross-domain ensemble (if applicable)
        if mode == "both" and len(results_by_domain["text"]) > 0 and len(results_by_domain["image"]) > 0:
            log.info(f"\n   🔀 Creating cross-domain ensemble for {task}...")
            
            # Test different alpha values
            for alpha in [0.25, 0.5, 0.75]:
                cross_result = best_two_domains(
                    task=task,
                    text_results=results_by_domain["text"],
                    image_results=results_by_domain["image"],
                    gold_df=gold,
                    alpha=alpha
                )
                
                if cross_result:
                    cross_result["domain"] = "ensemble"
                    ensemble_results.append(cross_result)
                    log.info(f"      ✅ Alpha={alpha}: F1={cross_result['F1']:.3f}")
    
    # Add ensemble results
    results.extend(ensemble_results)
    results_by_domain["ensemble"] = ensemble_results
    
    stage_time = time.time() - stage_start
    log.info(f"\n✅ Ensemble creation completed in {stage_time:.1f}s")
    log.info(f"   Ensembles created: {len(ensemble_results)}")
    
    # Save checkpoint
    save_pipeline_state("ensemble_creation", {
        "ensemble_count": len(ensemble_results),
        "total_models": len(results)
    })
    
    pipeline_progress.update(1)
    
    # =========================================================================
    # STAGE 6: EVALUATION AND EXPORT
    # =========================================================================
    pipeline_progress.set_description("🔬 Pipeline: Evaluation & Export")
    stage_start = time.time()
    
    log.info("\n" + "="*60)
    log.info("STAGE 6: EVALUATION AND EXPORT")
    log.info("="*60)
    
    if results:
        # 1. Export all results to CSV
        log.info("\n📊 Exporting results...")
        aggregate_results_across_domains(results_by_domain)
        
        # 2. Export ensemble-specific metrics
        if ensemble_results:
            export_ensemble_metrics(ensemble_results)
        
        # 3. Generate evaluation plots
        log.info("\n📈 Generating evaluation plots...")
        export_eval_plots(results, gold)
        
        # 4. Save best models
        log.info("\n💾 Saving best models...")
        best_models = {}
        
        for task in ["keto", "vegan"]:
            task_results = [r for r in results if r["task"] == task]
            if task_results:
                best = max(task_results, key=lambda x: x["F1"])
                
                # Get model object
                if "ensemble_model" in best:
                    best_models[task] = best["ensemble_model"]
                elif "model_object" in best:
                    best_models[task] = best["model_object"]
                
                log.info(f"   ✅ Best {task}: {best['model']} (F1={best['F1']:.3f})")
        
        if best_models:
            save_models_optimized(best_models, vectorizer, CFG.artifacts_dir)
        
        # 5. Display final results
        log.info("\n📋 FINAL RESULTS SUMMARY")
        table("All Models", results)
        
        # 6. Performance summary by domain
        log.info("\n📊 PERFORMANCE BY DOMAIN:")
        for domain, domain_results in results_by_domain.items():
            if domain_results:
                avg_f1 = np.mean([r["F1"] for r in domain_results])
                best_f1 = max(r["F1"] for r in domain_results)
                log.info(f"   {domain:>8}: Avg F1={avg_f1:.3f}, Best F1={best_f1:.3f}")
    
    else:
        log.warning("⚠️  No results to export")
    
    stage_time = time.time() - stage_start
    log.info(f"\n✅ Evaluation completed in {stage_time:.1f}s")
    
    pipeline_progress.update(1)
    
    # =========================================================================
    # PIPELINE COMPLETION
    # =========================================================================
    total_time = time.time() - pipeline_start
    
    log.info("\n" + "="*80)
    log.info("🏁 PIPELINE COMPLETE")
    log.info("="*80)
    log.info(f"   ├─ Total runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    log.info(f"   ├─ Mode: {mode}")
    log.info(f"   ├─ Models trained: {len(results)}")
    log.info(f"   └─ Pipeline stages: 6/6 completed")
    
    # Best model summary
    if results:
        log.info(f"\n🏆 BEST MODELS:")
        for task in ["keto", "vegan"]:
            task_results = [r for r in results if r["task"] == task]
            if task_results:
                best = max(task_results, key=lambda x: x["F1"])
                log.info(f"   {task:>5}: {best['model']} (F1={best['F1']:.3f}, ACC={best['ACC']:.3f})")
    
    # Memory summary
    final_memory = psutil.virtual_memory()
    log.info(f"\n💾 RESOURCE USAGE:")
    log.info(f"   ├─ Peak memory: {final_memory.percent:.1f}%")
    log.info(f"   ├─ Final memory: {final_memory.used // (1024**2)} MB")
    log.info(f"   └─ Models/second: {len(results)/total_time:.2f}")
    
    # Save final pipeline state
    save_pipeline_state("completed", {
        "total_time": total_time,
        "total_models": len(results),
        "best_models": best_models if 'best_models' in locals() else {},
        "pipeline_complete": True
    })
    
    return vectorizer, silver_txt, gold, results


