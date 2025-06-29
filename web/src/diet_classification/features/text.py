#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text feature extraction using TF-IDF vectorization.

This module handles the text feature extraction logic that was previously
embedded in run_full_pipeline around lines 7592-7641.
"""

from pathlib import Path
from typing import Tuple, Optional
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from ..core import log
from ..config import CFG
from ..data.usda import label_usda_keto_data, _load_usda_carb_table


def create_tfidf_vectorizer(**kwargs) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with default configuration.
    
    Args:
        **kwargs: Override default vectorizer parameters
        
    Returns:
        Configured TfidfVectorizer instance
    """
    default_kwargs = CFG.vec_kwargs.copy()
    default_kwargs.update(kwargs)
    
    log.info(f"   ├─ Vectorizer config: {default_kwargs}")
    return TfidfVectorizer(**default_kwargs)


def extract_text_features(
    silver_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
    extend_with_usda: bool = True
) -> Tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
    """
    Extract TF-IDF features from text data.
    
    Args:
        silver_df: Silver training data with 'clean' column
        gold_df: Gold test data with 'clean' column
        vectorizer: Pre-fitted vectorizer (if None, creates new one)
        extend_with_usda: Whether to extend silver data with USDA examples
        
    Returns:
        Tuple of (silver_features, gold_features, vectorizer)
    """
    log.info("🔤 Extracting text features...")
    
    # Extend silver data with USDA if requested
    if extend_with_usda:
        carb_df = _load_usda_carb_table()
        if not carb_df.empty:
            usda_labeled = label_usda_keto_data(carb_df)
            log.info(f"   ├─ USDA examples added: {len(usda_labeled)}")
            silver_df = pd.concat([silver_df, usda_labeled], ignore_index=True)
            
            # Save extended data
            Path("artifacts").mkdir(exist_ok=True)
            silver_df.to_csv("artifacts/silver_extended.csv", index=False)
        else:
            log.warning("   ├─ No USDA data added - carb_df is empty")
    
    # Create or use provided vectorizer
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer()
        log.info("   ├─ Fitting vectorizer on silver data...")
        X_text_silver = vectorizer.fit_transform(silver_df.clean)
    else:
        log.info("   ├─ Using pre-fitted vectorizer")
        X_text_silver = vectorizer.transform(silver_df.clean)
    
    # Transform gold data
    log.info("   ├─ Transforming gold data...")
    X_text_gold = vectorizer.transform(gold_df.clean)
    
    # Log statistics
    log.info(f"   📊 Text Feature Statistics:")
    log.info(f"   ├─ Vocabulary size: {len(vectorizer.vocabulary_):,}")
    log.info(f"   ├─ Silver features: {X_text_silver.shape}")
    log.info(f"   ├─ Gold features: {X_text_gold.shape}")
    log.info(f"   ├─ Sparsity: {(1 - X_text_silver.nnz / (X_text_silver.shape[0] * X_text_silver.shape[1])):.1%}")
    log.info(f"   └─ Memory usage: ~{X_text_silver.data.nbytes // (1024**2)} MB")
    
    # Save embeddings
    Path("embeddings").mkdir(exist_ok=True)
    joblib.dump(X_text_gold, "embeddings/text_gold.pkl")
    log.info("   ✅ Saved gold text embeddings")
    
    return X_text_silver, X_text_gold, vectorizer