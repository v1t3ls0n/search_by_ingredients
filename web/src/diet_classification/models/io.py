#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model saving and loading functions.

Based on original lines 822-897 and scattered model I/O logic from diet_classifiers.py
"""

import pickle
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any

import joblib

from ..core import log


def save_models_optimized(models: Dict[str, Any], vectorizer: Any, path: Path):
    """
    Save models with compression, versioning, and metadata.

    Args:
        models: Dictionary of task -> model
        vectorizer: TF-IDF vectorizer
        path: Directory to save models
        
    Based on original lines 822-897
    """
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
            n_features = model.n_features_in_
            if n_features <= 2048:
                model_info['requires_images'] = True  # Image-only model
                model_info['type'] = 'image'
            elif n_features > 7000:  # Combined features
                model_info['requires_images'] = True
                model_info['type'] = 'both'
            else:
                model_info['requires_images'] = False  # Text-only model
                model_info['type'] = 'text'
            model_info['feature_count'] = n_features

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


def load_models(vec_path: Path, models_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load models and vectorizer from disk.
    
    Args:
        vec_path: Path to vectorizer file
        models_path: Path to models file
        
    Returns:
        Tuple of (vectorizer, models_dict)
    """
    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    return vectorizer, models