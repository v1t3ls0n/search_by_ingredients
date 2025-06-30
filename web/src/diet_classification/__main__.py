#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the diet classification pipeline.

Based on original lines 8512-9195 from diet_classifiers.py
"""

import argparse
import json
import os
import sys
import atexit
from pathlib import Path

# Import components from our modular structure
from .core import log, get_pipeline_state
from .config import CFG
from .classification.keto import is_keto
from .classification.vegan import is_vegan
from .utils.memory import get_available_memory
from .utils.validation import preflight_checks
from .pipeline.orchestrator import run_full_pipeline
from .pipeline.evaluation import evaluate_ground_truth

import pandas as pd
import numpy as np


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
    
    Based on original lines 8512-9195 (simplified for clarity)
    """
    
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
                        default=None, help='Feature mode (defaults to "both" for training, "text" for evaluation)')
    parser.add_argument('--inference_mode', choices=['text', 'image', 'both'],
                        default=None, help='Inference mode for ingredient classification')
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
            sys.setswitchinterval(0.1)

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

            # Use mode-specific models if specified
            inference_mode = args.inference_mode or args.mode
            
            keto = is_keto(ingredients, mode=inference_mode)
            vegan = is_vegan(ingredients, mode=inference_mode)
            
            result = {
                'keto': keto, 
                'vegan': vegan,
                'inference_mode': inference_mode or 'auto'
            }
            
            log.info(json.dumps(result))
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
                from .pipeline.checkpoints import load_pipeline_state
                saved_stage, saved_data = load_pipeline_state()
                resume_from_checkpoint = False
                
                if saved_stage and not args.force:
                    log.info(f"   📂 Found saved pipeline state from stage: {saved_stage}")
                    response = input("   Resume from saved state? [Y/n]: ").strip().lower()
                    
                    if response != 'n':
                        resume_from_checkpoint = True
                        log.info(f"   ✅ Will attempt to resume from {saved_stage}")
                        
                        # Load any saved models into state
                        pipeline_state = get_pipeline_state()
                        if 'best_models' in saved_data:
                            pipeline_state.best_models.update(saved_data['best_models'])
                            log.info(f"   ├─ Restored {len(saved_data['best_models'])} models to cache")
                
                # Run pipeline (it will handle resume internally)
                results = run_full_pipeline(
                    mode=args.mode or 'both', 
                    force=args.force, 
                    sample_frac=args.sample_frac
                )
                
                # Unpack results
                if len(results) == 5:
                    vec, silver, gold, res, model_registry = results
                else:
                    vec, silver, gold, res = results
                    model_registry = None
                
                if not res:
                    log.error("❌ Pipeline produced no results!")
                    sys.exit(1)

                log.info(f"✅ Pipeline completed with {len(res)} results")

                # Save models with optimized serialization
                try:
                    from .models.io import save_models_optimized, save_best_models_by_domain
                    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save best models by domain if not already done
                    if model_registry is None:
                        model_registry = save_best_models_by_domain(res, vec, CFG.artifacts_dir)
                    
                    # Prepare best overall models
                    best_models = {}
                    for task in ['keto', 'vegan']:
                        task_res = [r for r in res if r['task'] == task]
                        if task_res:
                            best = max(task_res, key=lambda x: x['F1'])
                            
                            if 'ensemble_model' in best and best['ensemble_model'] is not None:
                                best_models[task] = best['ensemble_model']
                                log.info(f"✅ Selected {task} ensemble: {best['model']} (F1={best['F1']:.3f})")
                            else:
                                # Fallback to single model from state
                                pipeline_state = get_pipeline_state()
                                model_name = best['model']
                                base_name = model_name.split('_')[0]
                                if base_name in pipeline_state.best_models:
                                    best_models[task] = pipeline_state.best_models[base_name]
                                    log.info(f"✅ Selected {task} model: {base_name} (F1={best['F1']:.3f})")
                    
                    if best_models:
                        # Use optimized saving
                        save_models_optimized(best_models, vec, CFG.artifacts_dir)
                        
                        # Save final pipeline state
                        from .pipeline.checkpoints import save_pipeline_state
                        final_state = {
                            'stage': 'completed',
                            'best_models': {k: v for k, v in pipeline_state.best_models.items()},
                            'model_registry': model_registry,
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
            
            # Default to text mode for ground truth evaluation if not explicitly set
            if args.mode is None:
                args.mode = 'text'
                log.info(f"   ℹ️  Defaulting to text-only evaluation (use --mode both for all features)")
            
            evaluate_ground_truth(args.ground_truth, mode=args.mode)

        elif args.predict:
            log.info(f"🔮 Running prediction on unlabeled data: {args.predict}")
            
            # Simplified batch prediction logic
            from .pipeline.prediction import batch_predict
            batch_predict(args.predict)

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