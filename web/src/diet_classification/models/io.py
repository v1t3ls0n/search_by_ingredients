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
from typing import Dict, Tuple, Any, List, Optional

import joblib

from ..core import log, get_pipeline_state
from ..config import CFG


def save_models_optimized(models: Dict[str, Any], vectorizer: Any, path: Path):
    """
    Save models with compression, versioning, and metadata.
    
    Enhanced to save best models by domain type.

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
        'models': {},
        'best_models_by_domain': {}  # New: track best models by domain
    }

    # Save each model with compression
    for task, model in models.items():
        model_path = path / f"model_{task}.pkl.gz"

        # Determine model requirements and domain type
        model_info = {
            'task': task,
            'model_type': type(model).__name__,
            'requires_images': False,
            'feature_count': None,
            'domain': None  # New: track domain
        }

        # Determine domain based on model characteristics
        if hasattr(model, 'ensemble_type'):
            model_info['ensemble_type'] = model.ensemble_type
            model_info['requires_images'] = model.ensemble_type in [
                'best_two', 'smart_ensemble']
            model_info['domain'] = 'ensemble'
        elif hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            if n_features <= 2048:
                model_info['requires_images'] = True  # Image-only model
                model_info['domain'] = 'image'
            elif n_features > 7000:  # Combined features
                model_info['requires_images'] = True
                model_info['domain'] = 'both'
            else:
                model_info['requires_images'] = False  # Text-only model
                model_info['domain'] = 'text'
            model_info['feature_count'] = n_features
        else:
            # Default to text if we can't determine
            model_info['domain'] = 'text'

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
        # Also ensure pipeline state has the model
        pipeline_state = get_pipeline_state()
        pipeline_state.set_best_model(task, model)

    with open(path / "models.pkl", 'wb') as f:
        pickle.dump(compat_models, f)

    with open(path / "vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)

    log.info(f"   ✅ Created backward-compatible model files")


def save_best_models_by_domain(
    results: List[Dict[str, Any]], 
    vectorizer: Any,
    path: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Save the best model for each task and domain combination.
    
    This creates a comprehensive model library with:
    - Best text model for keto
    - Best text model for vegan
    - Best image model for keto
    - Best image model for vegan
    - Best hybrid model for keto
    - Best hybrid model for vegan
    - Best ensemble for keto
    - Best ensemble for vegan
    
    Args:
        results: List of all training results
        vectorizer: TF-IDF vectorizer
        path: Directory to save models
        
    Returns:
        Model registry dictionary
    """
    path.mkdir(parents=True, exist_ok=True)
    
    # Organize results by task and domain
    organized_results = {}
    for result in results:
        task = result.get('task')
        domain = result.get('domain', 'unknown')
        
        if task not in organized_results:
            organized_results[task] = {}
        
        if domain not in organized_results[task]:
            organized_results[task][domain] = []
            
        organized_results[task][domain].append(result)
    
    # Find and save best models
    best_models = {}
    model_registry = {}
    
    for task in ['keto', 'vegan']:
        if task not in organized_results:
            continue
            
        best_models[task] = {}
        
        for domain in ['text', 'image', 'both', 'ensemble']:
            if domain not in organized_results[task]:
                continue
                
            # Find best model for this task/domain combination
            domain_results = organized_results[task][domain]
            if domain_results:
                best_result = max(domain_results, key=lambda x: x.get('F1', 0))
                
                # Get model object
                model_obj = None
                if 'ensemble_model' in best_result:
                    model_obj = best_result['ensemble_model']
                elif 'model_object' in best_result:
                    model_obj = best_result['model_object']
                
                if model_obj:
                    model_key = f"{task}_{domain}"
                    best_models[task][domain] = model_obj
                    model_registry[model_key] = {
                        'model': model_obj,
                        'metrics': {
                            'F1': best_result.get('F1', 0),
                            'ACC': best_result.get('ACC', 0),
                            'PREC': best_result.get('PREC', 0),
                            'REC': best_result.get('REC', 0)
                        },
                        'model_name': best_result.get('model', 'unknown')
                    }
                    
                    # Save individual model
                    model_path = path / f"best_{model_key}.pkl.gz"
                    joblib.dump(model_obj, model_path, compress=3)
                    log.info(f"   ✅ Saved best {task} {domain} model: {best_result.get('model')} (F1={best_result.get('F1', 0):.3f})")
    
    # Save complete model registry
    registry_path = path / "model_registry.pkl"
    with open(registry_path, 'wb') as f:
        pickle.dump(model_registry, f)
    
    # Save metadata
    metadata = {
        'creation_time': datetime.now().isoformat(),
        'models_by_domain': {}
    }
    
    for key, info in model_registry.items():
        metadata['models_by_domain'][key] = {
            'model_name': info['model_name'],
            'metrics': info['metrics']
        }
    
    metadata_path = path / "best_models_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save vectorizer
    vec_path = path / "vectorizer.pkl.gz"
    joblib.dump(vectorizer, vec_path, compress=3)
    
    # Update pipeline state
    pipeline_state = get_pipeline_state()
    for key, info in model_registry.items():
        pipeline_state.models[key] = info['model']
    
    log.info(f"\n   📊 Model Registry Summary:")
    for task in ['keto', 'vegan']:
        if task in best_models:
            log.info(f"   {task.upper()}:")
            for domain in ['text', 'image', 'both', 'ensemble']:
                if domain in best_models[task]:
                    key = f"{task}_{domain}"
                    if key in model_registry:
                        info = model_registry[key]
                        log.info(f"      ├─ {domain}: {info['model_name']} (F1={info['metrics']['F1']:.3f})")
    
    return model_registry


def load_models(vec_path: Path, models_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Load models and vectorizer from disk.
    
    Enhanced to handle domain-specific models.
    
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
    
    # Check if we have a model registry
    registry_path = models_path.parent / "model_registry.pkl"
    if registry_path.exists():
        try:
            with open(registry_path, 'rb') as f:
                model_registry = pickle.load(f)
            
            # Add registry models to the models dict
            for key, info in model_registry.items():
                models[key] = info['model']
            
            log.info(f"   ✅ Loaded model registry with {len(model_registry)} models")
        except Exception as e:
            log.warning(f"   ⚠️  Could not load model registry: {e}")
    
    # Ensure we have simple task keys for backward compatibility
    if 'keto' not in models:
        # Find best keto model (prefer ensemble, then both, then text)
        for domain in ['ensemble', 'both', 'text', 'image']:
            key = f"keto_{domain}"
            if key in models:
                models['keto'] = models[key]
                break
            # Also check old format
            keto_models = {k: v for k, v in models.items() if 'keto' in k.lower()}
            if keto_models:
                # Prefer ensemble, then both, then text
                for key_type in ['ensemble', 'both', 'text']:
                    matching = [k for k in keto_models.keys() if key_type in k.lower()]
                    if matching:
                        models['keto'] = keto_models[matching[0]]
                        break
                else:
                    # Just use first one if no preferences match
                    models['keto'] = list(keto_models.values())[0]
    
    if 'vegan' not in models:
        # Find best vegan model
        for domain in ['ensemble', 'both', 'text', 'image']:
            key = f"vegan_{domain}"
            if key in models:
                models['vegan'] = models[key]
                break
            # Also check old format
            vegan_models = {k: v for k, v in models.items() if 'vegan' in k.lower()}
            if vegan_models:
                # Prefer ensemble, then both, then text
                for key_type in ['ensemble', 'both', 'text']:
                    matching = [k for k in vegan_models.keys() if key_type in k.lower()]
                    if matching:
                        models['vegan'] = vegan_models[matching[0]]
                        break
                else:
                    # Just use first one if no preferences match
                    models['vegan'] = list(vegan_models.values())[0]
    
    # Update pipeline state
    pipeline_state = get_pipeline_state()
    for key, model in models.items():
        if '_' in key:
            # Domain-specific model
            pipeline_state.models[key] = model
        else:
            # Task-only model
            pipeline_state.set_best_model(key, model)
    
    return vectorizer, models


def load_best_model_for_mode(task: str, mode: str) -> Optional[Any]:
    """
    Load the best model for a specific task and mode.
    
    Args:
        task: 'keto' or 'vegan'
        mode: 'text', 'image', or 'both'
        
    Returns:
        Best model for the task/mode combination or None
    """
    pipeline_state = get_pipeline_state()
    
    # First check if models are already loaded
    model_key = f"{task}_{mode}"
    if model_key in pipeline_state.models:
        return pipeline_state.models[model_key]
    
    # Try to load from disk
    for directory in [CFG.artifacts_dir, CFG.pretrained_models_dir]:
        model_path = directory / f"best_{model_key}.pkl.gz"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                pipeline_state.models[model_key] = model
                log.info(f"   ✅ Loaded {task} {mode} model from {model_path}")
                return model
            except Exception as e:
                log.warning(f"   ⚠️  Failed to load {model_path}: {e}")
    
    # Fall back to general model
    return pipeline_state.get_best_model(task)