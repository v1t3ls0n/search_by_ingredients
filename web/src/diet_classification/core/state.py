#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized state management for the diet classification pipeline.

Based on original lines 345-426 from diet_classifiers.py
This replaces all the global variables with a proper singleton pattern.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

# Module-level singleton instance
_instance = None


class PipelineStateManager:
    """
    Centralized state management for the pipeline.
    
    This singleton class replaces all the global variables from the original code:
    - _DATASETS -> self.datasets
    - _CARB_MAP -> self.carb_map
    - _pipeline_state -> self.models, self.vectorizer
    - BEST -> self.best_models
    """

    def __init__(self):
        # Import here to avoid circular dependency
        from .logging import get_logger
        from ..config import CFG
        
        self.log = get_logger()
        self.cfg = CFG
        
        # Dataset caches (replaces _DATASETS global)
        self.datasets = None
        
        # USDA carb lookup (replaces _CARB_MAP and _FUZZY_KEYS globals)
        self.carb_map = None
        self.fuzzy_keys = None
        
        # ML pipeline state (replaces _pipeline_state global)
        self.vectorizer = None
        self.models = {}
        self.initialized = False
        
        # Best models cache (replaces BEST global)
        self.best_models = {}
        
        # Checkpoint management
        self.checkpoints = {}
        
        # Memory management
        self.memory_mode = None  # 'high', 'medium', 'low'
        
        # Update memory mode on initialization
        self.update_memory_mode()

    def update_memory_mode(self):
        """Update memory mode based on available resources."""
        # Import here to avoid circular dependency
        from ..utils.memory import get_available_memory
        
        available_gb = get_available_memory()

        if available_gb >= self.cfg.memory_thresholds['high']:
            self.memory_mode = 'high'
        elif available_gb >= self.cfg.memory_thresholds['medium']:
            self.memory_mode = 'medium'
        else:
            self.memory_mode = 'low'

        self.log.info(
            f"Memory mode set to: {self.memory_mode} ({available_gb:.1f} GB available)")
        return self.memory_mode

    def should_use_memmap(self, data_size: int) -> bool:
        """Determine if memory-mapped arrays should be used."""
        return (data_size > self.cfg.memory_thresholds['memmap'] or 
                self.memory_mode == 'low')

    def save_checkpoint(self, stage: str, data: dict):
        """Save a checkpoint for the given stage."""
        checkpoint_path = self.cfg.checkpoints_dir / f"checkpoint_{stage}.pkl"
        self.checkpoints[stage] = data

        checkpoint_data = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'memory_mode': self.memory_mode,
            'pipeline_version': '1.0',
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        self.log.info(f"Saved checkpoint: {stage}")

    def load_checkpoint(self, stage: str):
        """Load a checkpoint for the given stage."""
        checkpoint_path = self.cfg.checkpoints_dir / f"checkpoint_{stage}.pkl"

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)

                self.checkpoints[stage] = checkpoint_data['data']
                self.log.info(
                    f"Loaded checkpoint: {stage} (saved at {checkpoint_data['timestamp']})")
                return checkpoint_data['data']
            except Exception as e:
                self.log.error(f"Failed to load checkpoint {stage}: {e}")

        return None

    def clear(self):
        """Clear all state."""
        self.datasets = None
        self.carb_map = None
        self.fuzzy_keys = None
        self.vectorizer = None
        self.models.clear()
        self.best_models.clear()
        self.checkpoints.clear()
        self.initialized = False
        
    def ensure_pipeline_initialized(self):
        """
        Ensure the ML pipeline is initialized.
        
        This replaces the _ensure_pipeline() function from the original code.
        """
        if self.initialized:
            return
            
        # Import here to avoid circular dependency
        from ..models.io import load_models
        
        vec_path = self.cfg.artifacts_dir / "vectorizer.pkl"
        models_path = self.cfg.artifacts_dir / "models.pkl"
        
        try:
            # Attempt to load existing models
            if vec_path.exists() and models_path.exists():
                self.vectorizer, self.models = load_models(vec_path, models_path)
                self.log.info("Loaded vectorizer + models from %s", self.cfg.artifacts_dir)
                self.initialized = True
            else:
                self.log.info("No saved artifacts found - models need to be trained")
                # Don't set initialized=True here, let training pipeline handle it
                
        except Exception as e:
            self.log.warning("Model loading failed (%s). Need to train models.", e)
            # Don't set initialized=True on failure


def get_pipeline_state() -> PipelineStateManager:
    """
    Get the singleton pipeline state instance.
    
    This function ensures only one instance of PipelineStateManager exists
    throughout the application lifecycle.
    
    Returns:
        The singleton PipelineStateManager instance
    """
    global _instance
    if _instance is None:
        _instance = PipelineStateManager()
    return _instance