#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline prediction functions for batch processing.

This module provides batch prediction capabilities for processing
large datasets of unlabeled ingredients.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
import time
from tqdm import tqdm

from ..core import log, get_pipeline_state
from ..config import CFG
from ..classification.keto import is_keto
from ..classification.vegan import is_vegan
from ..data.preprocessing import normalise


def batch_predict(input_path: str, output_path: Optional[str] = None) -> None:
    """
    Run batch prediction on a CSV file of ingredients.
    
    Expected CSV format:
    - Must have 'ingredients' column
    - Optional 'id' or 'recipe_id' column for tracking
    - Any other columns will be preserved in output
    
    Args:
        input_path: Path to input CSV file
        output_path: Path for output CSV (if None, auto-generated)
    """
    log.info("🔮 BATCH PREDICTION PIPELINE")
    
    try:
        # Load input data
        log.info(f"   ├─ Loading input data: {input_path}")
        df = pd.read_csv(input_path)
        
        # Validate input
        if 'ingredients' not in df.columns:
            log.error(f"   ❌ Missing required 'ingredients' column")
            log.info(f"   ℹ️  Available columns: {list(df.columns)}")
            return
        
        log.info(f"   ├─ Dataset size: {len(df)} recipes")
        log.info(f"   ├─ Columns: {list(df.columns)}")
        
        # Check for null ingredients
        null_count = df['ingredients'].isnull().sum()
        if null_count > 0:
            log.warning(f"   ⚠️  Found {null_count} null ingredients, will skip these")
            df = df.dropna(subset=['ingredients']).copy()
            log.info(f"   ├─ Processing {len(df)} non-null recipes")
        
        # Prepare output dataframe
        results_df = df.copy()
        results_df['clean_ingredients'] = results_df['ingredients'].map(normalise)
        
        # Initialize prediction columns
        results_df['is_keto'] = False
        results_df['is_vegan'] = False
        results_df['prediction_time'] = 0.0
        results_df['prediction_error'] = ''
        
        # Run predictions with progress bar
        log.info(f"   🧠 Running predictions...")
        
        predictions_completed = 0
        predictions_failed = 0
        
        for idx, row in tqdm(results_df.iterrows(), 
                           total=len(results_df), 
                           desc="   ├─ Predicting",
                           bar_format="   ├─ {desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            
            start_time = time.time()
            error_msg = ''
            
            try:
                ingredients = row['ingredients']
                
                # Make predictions
                keto_pred = is_keto(ingredients)
                vegan_pred = is_vegan(ingredients)
                
                # Store results
                results_df.at[idx, 'is_keto'] = keto_pred
                results_df.at[idx, 'is_vegan'] = vegan_pred
                
                predictions_completed += 1
                
            except Exception as e:
                error_msg = str(e)[:100]  # Truncate long error messages
                predictions_failed += 1
                log.debug(f"   ├─ Prediction failed for row {idx}: {e}")
                
                # Set default values on error
                results_df.at[idx, 'is_keto'] = False
                results_df.at[idx, 'is_vegan'] = True  # Default to vegan when unsure
            
            # Record timing and errors
            prediction_time = time.time() - start_time
            results_df.at[idx, 'prediction_time'] = prediction_time
            results_df.at[idx, 'prediction_error'] = error_msg
        
        # Generate output path if not provided
        if output_path is None:
            input_stem = Path(input_path).stem
            output_path = f"{input_stem}_predictions.csv"
        
        # Save results
        results_df.to_csv(output_path, index=False)
        
        # Calculate summary statistics
        total_predictions = len(results_df)
        avg_time = results_df['prediction_time'].mean()
        keto_count = results_df['is_keto'].sum()
        vegan_count = results_df['is_vegan'].sum()
        both_count = (results_df['is_keto'] & results_df['is_vegan']).sum()
        
        # Log summary
        log.info(f"   📊 PREDICTION SUMMARY:")
        log.info(f"   ├─ Total predictions: {total_predictions}")
        log.info(f"   ├─ Successful: {predictions_completed}")
        log.info(f"   ├─ Failed: {predictions_failed}")
        log.info(f"   ├─ Average time per prediction: {avg_time:.3f}s")
        log.info(f"   ├─ Keto recipes: {keto_count} ({keto_count/total_predictions:.1%})")
        log.info(f"   ├─ Vegan recipes: {vegan_count} ({vegan_count/total_predictions:.1%})")
        log.info(f"   ├─ Both keto & vegan: {both_count} ({both_count/total_predictions:.1%})")
        log.info(f"   └─ Results saved to: {output_path}")
        
        # Save summary report
        summary_path = Path(output_path).parent / f"{Path(output_path).stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("BATCH PREDICTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input file: {input_path}\n")
            f.write(f"Output file: {output_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total recipes processed: {total_predictions}\n")
            f.write(f"Successful predictions: {predictions_completed}\n")
            f.write(f"Failed predictions: {predictions_failed}\n")
            f.write(f"Average prediction time: {avg_time:.3f} seconds\n\n")
            f.write(f"Classification Results:\n")
            f.write(f"  Keto-friendly: {keto_count} ({keto_count/total_predictions:.1%})\n")
            f.write(f"  Vegan: {vegan_count} ({vegan_count/total_predictions:.1%})\n")
            f.write(f"  Both keto & vegan: {both_count} ({both_count/total_predictions:.1%})\n")
            
            if predictions_failed > 0:
                f.write(f"\nErrors occurred in {predictions_failed} predictions.\n")
                f.write(f"Check the 'prediction_error' column in the output file for details.\n")
        
        log.info(f"   📄 Summary report saved to: {summary_path}")
        
    except FileNotFoundError:
        log.error(f"   ❌ Input file not found: {input_path}")
    except Exception as e:
        log.error(f"   ❌ Batch prediction failed: {e}")
        import traceback
        log.debug(f"Full traceback:\n{traceback.format_exc()}")


def predict_single_ingredient(ingredient: str) -> dict:
    """
    Predict diet classifications for a single ingredient.
    
    Args:
        ingredient: Raw ingredient string
        
    Returns:
        Dictionary with prediction results
    """
    start_time = time.time()
    
    try:
        keto_pred = is_keto([ingredient])
        vegan_pred = is_vegan([ingredient])
        
        prediction_time = time.time() - start_time
        
        return {
            'ingredient': ingredient,
            'clean_ingredient': normalise(ingredient),
            'is_keto': keto_pred,
            'is_vegan': vegan_pred,
            'prediction_time': prediction_time,
            'error': None
        }
        
    except Exception as e:
        prediction_time = time.time() - start_time
        return {
            'ingredient': ingredient,
            'clean_ingredient': normalise(ingredient),
            'is_keto': False,
            'is_vegan': True,  # Default to vegan when unsure
            'prediction_time': prediction_time,
            'error': str(e)
        }


def predict_ingredient_list(ingredients: List[str]) -> List[dict]:
    """
    Predict diet classifications for a list of ingredients.
    
    Args:
        ingredients: List of ingredient strings
        
    Returns:
        List of prediction result dictionaries
    """
    log.info(f"🔮 Predicting classifications for {len(ingredients)} ingredients")
    
    results = []
    for ingredient in tqdm(ingredients, desc="Predicting"):
        result = predict_single_ingredient(ingredient)
        results.append(result)
    
    # Summary
    keto_count = sum(1 for r in results if r['is_keto'])
    vegan_count = sum(1 for r in results if r['is_vegan'])
    error_count = sum(1 for r in results if r['error'] is not None)
    
    log.info(f"   ✅ Predictions complete:")
    log.info(f"   ├─ Keto: {keto_count}/{len(ingredients)} ({keto_count/len(ingredients):.1%})")
    log.info(f"   ├─ Vegan: {vegan_count}/{len(ingredients)} ({vegan_count/len(ingredients):.1%})")
    log.info(f"   └─ Errors: {error_count}/{len(ingredients)}")
    
    return results