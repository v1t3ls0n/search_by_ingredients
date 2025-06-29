#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature combination utilities for multi-modal learning.

Based on original lines 3376-3391, 3499-3523, 3540-3564 from diet_classifiers.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

from ..core import log


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
        
    Based on original lines 3376-3391
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
        
    Based on original lines 3499-3523
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
    
    Based on original lines 3540-3564
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


def apply_smote(X, y, max_dense_size: int = None):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) for class balancing.
    
    Based on original lines 3587-3632
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from ..config import CFG
    
    if max_dense_size is None:
        max_dense_size = CFG.model_config['smote_max_dense_size']
    
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