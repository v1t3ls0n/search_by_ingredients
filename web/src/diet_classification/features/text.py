#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text feature extraction module for diet classification.

This module implements TF-IDF vectorization with USDA nutritional data
integration for enhanced feature representation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core import log, get_pipeline_state
from ..config import CFG
from ..data.preprocessing import normalise
from ..data.usda import carbs_per_100g, label_usda_keto_data, ensure_carb_map
from ..utils.memory import optimize_memory_usage


def create_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    """
    Create a configured TF-IDF vectorizer.
    
    Uses configuration from CFG.vec_kwargs by default, but allows overrides.
    
    Args:
        **kwargs: Override parameters for TfidfVectorizer
        
    Returns:
        Configured TfidfVectorizer instance
    """
    # Start with config defaults
    vec_params = CFG.vec_kwargs.copy()
    
    # Apply any overrides
    vec_params.update(kwargs)
    
    log.info(f"Creating TF-IDF vectorizer with params: {vec_params}")
    
    return TfidfVectorizer(**vec_params)


def extract_text_features(
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
    include_usda: bool = True,
    sample_frac: Optional[float] = None
) -> Tuple[csr_matrix, csr_matrix, TfidfVectorizer, pd.DataFrame]:
    """
    Extract TF-IDF features from text data with optional USDA integration.
    
    This function implements the complete text feature extraction pipeline:
    1. Creates or uses provided TF-IDF vectorizer
    2. Optionally integrates USDA nutritional data for enhanced features
    3. Fits vectorizer on silver data (train set)
    4. Transforms both silver and gold data
    5. Returns sparse matrices for memory efficiency
    
    Args:
        silver_df: Training DataFrame with 'clean' column
        gold_df: Test DataFrame with 'clean' column
        vectorizer: Optional pre-fitted vectorizer (creates new if None)
        include_usda: Whether to integrate USDA nutritional data
        sample_frac: Optional sampling fraction for silver data
        
    Returns:
        Tuple of (X_silver, X_gold, vectorizer, silver_extended)
        - X_silver: Sparse TF-IDF matrix for silver data
        - X_gold: Sparse TF-IDF matrix for gold data
        - vectorizer: Fitted TfidfVectorizer
        - silver_extended: Silver DataFrame potentially extended with USDA data
    """
    pipeline_state = get_pipeline_state()
    
    log.info("🔤 Extracting text features")
    log.info(f"   ├─ Silver samples: {len(silver_df):,}")
    log.info(f"   ├─ Gold samples: {len(gold_df):,}")
    log.info(f"   ├─ Include USDA: {include_usda}")
    log.info(f"   └─ Sample fraction: {sample_frac or 'Full dataset'}")
    
    # Apply sampling if requested
    if sample_frac and sample_frac < 1.0:
        original_size = len(silver_df)
        silver_df = silver_df.sample(frac=sample_frac, random_state=42).copy()
        log.info(f"   📉 Sampled silver: {original_size:,} → {len(silver_df):,} ({sample_frac:.1%})")
    
    # Ensure 'clean' column exists
    if 'clean' not in silver_df.columns:
        log.info("   🧹 Normalizing silver text...")
        silver_df['clean'] = silver_df['ingredients'].fillna("").map(normalise)
    
    if 'clean' not in gold_df.columns:
        log.info("   🧹 Normalizing gold text...")
        gold_df['clean'] = gold_df['ingredients'].fillna("").map(normalise)
    
    # Make a copy to avoid modifying original
    silver_extended = silver_df.copy()
    
    # Integrate USDA data if requested
    if include_usda:
        log.info("   🥗 Integrating USDA nutritional data...")
        
        try:
            # Ensure carb map is loaded
            ensure_carb_map()
            
            # Get USDA carb data from pipeline state
            if pipeline_state.carb_map:
                # Convert carb map to DataFrame
                carb_df = pd.DataFrame(
                    list(pipeline_state.carb_map.items()),
                    columns=['food_desc', 'carb_100g']
                )
                
                if not carb_df.empty:
                    # Convert to labeled training data
                    usda_labeled = label_usda_keto_data(carb_df)
                    
                    log.info(f"      ├─ USDA entries: {len(usda_labeled):,}")
                    log.info(f"      ├─ Keto positive: {usda_labeled['silver_keto'].sum():,}")
                    
                    # Concatenate with silver data
                    silver_extended = pd.concat(
                        [silver_extended, usda_labeled],
                        ignore_index=True
                    )
                    
                    log.info(f"      └─ Extended silver: {len(silver_extended):,} total")
                    
                    # Save extended dataset for debugging
                    try:
                        artifacts_path = CFG.artifacts_dir / "silver_extended.csv"
                        silver_extended.to_csv(artifacts_path, index=False)
                        log.debug(f"      💾 Saved extended silver to {artifacts_path}")
                    except Exception as e:
                        log.warning(f"      ⚠️  Could not save extended silver: {e}")
                else:
                    log.warning("      ⚠️  USDA carb data is empty")
            else:
                log.warning("      ⚠️  No USDA data in pipeline state")
                
        except Exception as e:
            log.error(f"   ❌ USDA integration failed: {e}")
            log.info("   ├─ Continuing with original silver data only")
    
    # Create or validate vectorizer
    if vectorizer is None:
        log.info("   🔧 Creating new TF-IDF vectorizer")
        vectorizer = create_tfidf_vectorizer()
        
        # Fit on extended silver data
        log.info(f"   📚 Fitting vectorizer on {len(silver_extended):,} samples")
        X_silver = vectorizer.fit_transform(silver_extended['clean'])
        
        # Store in pipeline state
        pipeline_state.vectorizer = vectorizer
        
    else:
        log.info("   ✅ Using provided vectorizer")
        
        # Just transform
        X_silver = vectorizer.transform(silver_extended['clean'])
    
    # Transform gold data
    log.info("   🔄 Transforming gold data")
    X_gold = vectorizer.transform(gold_df['clean'])
    
    # Log statistics
    vocab_size = len(vectorizer.vocabulary_)
    sparsity_silver = 1 - (X_silver.nnz / (X_silver.shape[0] * X_silver.shape[1]))
    sparsity_gold = 1 - (X_gold.nnz / (X_gold.shape[0] * X_gold.shape[1]))
    
    log.info(f"\n   📊 Text Feature Statistics:")
    log.info(f"   ├─ Vocabulary size: {vocab_size:,}")
    log.info(f"   ├─ Silver features: {X_silver.shape}")
    log.info(f"   ├─ Gold features: {X_gold.shape}")
    log.info(f"   ├─ Silver sparsity: {sparsity_silver:.1%}")
    log.info(f"   ├─ Gold sparsity: {sparsity_gold:.1%}")
    log.info(f"   └─ Memory usage: ~{X_silver.data.nbytes // (1024**2)} MB")
    
    # Memory optimization
    optimize_memory_usage("Text feature extraction")
    
    return X_silver, X_gold, vectorizer, silver_extended


def add_custom_features(
    X: csr_matrix,
    df: pd.DataFrame,
    feature_types: list[str] = None
) -> csr_matrix:
    """
    Add custom engineered features to the TF-IDF matrix.
    
    This function can add additional features like:
    - Ingredient count
    - Average word length
    - Numeric indicators (presence of numbers)
    - Special character counts
    
    Args:
        X: Sparse TF-IDF matrix
        df: DataFrame with original text
        feature_types: List of feature types to add
        
    Returns:
        Extended sparse matrix with additional features
    """
    if feature_types is None:
        feature_types = ['ingredient_count', 'avg_word_length']
    
    log.info(f"   🔧 Adding custom features: {feature_types}")
    
    custom_features = []
    
    if 'ingredient_count' in feature_types:
        # Count number of ingredients (commas + 1)
        ing_count = df['ingredients'].str.count(',') + 1
        custom_features.append(ing_count.values.reshape(-1, 1))
    
    if 'avg_word_length' in feature_types:
        # Average word length in clean text
        avg_len = df['clean'].str.split().apply(
            lambda x: np.mean([len(w) for w in x]) if x else 0
        )
        custom_features.append(avg_len.values.reshape(-1, 1))
    
    if 'has_numbers' in feature_types:
        # Binary indicator for presence of numbers
        has_nums = df['ingredients'].str.contains(r'\d', regex=True).astype(int)
        custom_features.append(has_nums.values.reshape(-1, 1))
    
    if custom_features:
        # Convert to sparse and concatenate
        custom_sparse = csr_matrix(np.hstack(custom_features))
        X_extended = hstack([X, custom_sparse])
        
        log.info(f"      ├─ Original shape: {X.shape}")
        log.info(f"      └─ Extended shape: {X_extended.shape}")
        
        return X_extended
    
    return X


def get_feature_importance(
    vectorizer: TfidfVectorizer,
    model,
    top_n: int = 20,
    task: str = "keto"
) -> pd.DataFrame:
    """
    Extract top features by importance from a trained model.
    
    Works with linear models that have coef_ attribute.
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        model: Trained model with coef_ attribute
        top_n: Number of top features to return
        task: Task name for labeling
        
    Returns:
        DataFrame with top features and their coefficients
    """
    if not hasattr(model, 'coef_'):
        log.warning(f"Model {type(model).__name__} doesn't have coef_ attribute")
        return pd.DataFrame()
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients
    coefs = model.coef_.ravel()
    
    # Create DataFrame
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs)
    })
    
    # Sort by absolute coefficient
    feature_df = feature_df.sort_values('abs_coefficient', ascending=False)
    
    # Get top positive and negative features
    top_positive = feature_df[feature_df['coefficient'] > 0].head(top_n)
    top_negative = feature_df[feature_df['coefficient'] < 0].head(top_n)
    
    log.info(f"\n   📊 Top {task} features:")
    log.info(f"   ├─ Positive indicators:")
    for _, row in top_positive.head(5).iterrows():
        log.info(f"   │  ├─ '{row['feature']}': {row['coefficient']:.3f}")
    
    log.info(f"   └─ Negative indicators:")
    for _, row in top_negative.head(5).iterrows():
        log.info(f"      ├─ '{row['feature']}': {row['coefficient']:.3f}")
    
    return pd.concat([top_positive, top_negative])


def save_vocabulary(vectorizer: TfidfVectorizer, output_path: str = None):
    """
    Save the vocabulary and IDF weights for analysis.
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        output_path: Path to save vocabulary (defaults to artifacts/)
    """
    if output_path is None:
        output_path = CFG.artifacts_dir / "vocabulary.csv"
    
    # Get vocabulary and IDF weights
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_
    
    # Create DataFrame
    vocab_df = pd.DataFrame([
        {'term': term, 'index': idx, 'idf': idf[idx]}
        for term, idx in vocab.items()
    ])
    
    # Sort by IDF (most distinctive terms first)
    vocab_df = vocab_df.sort_values('idf', ascending=False)
    
    # Save
    vocab_df.to_csv(output_path, index=False)
    log.info(f"   💾 Saved vocabulary ({len(vocab_df):,} terms) to {output_path}")
    
    # Log some statistics
    log.info(f"   📊 Vocabulary statistics:")
    log.info(f"   ├─ Total terms: {len(vocab_df):,}")
    log.info(f"   ├─ Max IDF: {vocab_df['idf'].max():.3f}")
    log.info(f"   ├─ Min IDF: {vocab_df['idf'].min():.3f}")
    log.info(f"   └─ Mean IDF: {vocab_df['idf'].mean():.3f}")