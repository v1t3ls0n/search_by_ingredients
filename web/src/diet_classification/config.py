#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for diet classification pipeline.

Centralized configuration with validation and automatic directory creation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Mapping
import logging


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration container for the pipeline with validation.
    
    All paths and settings are centralized here for easy management.
    """
    # Directory paths
    pretrained_models_dir: Path = Path("/app/pretrained_models")
    artifacts_dir: Path = Path("/app/artifacts")
    logs_dir: Path = Path("/app/artifacts/logs")
    data_dir: Path = Path("/app/data")
    usda_dir: Path = Path("/app/data/usda")
    state_dir: Path = Path("/app/pipeline_state")
    checkpoints_dir: Path = Path("/app/pipeline_state/checkpoints")
    cache_dir: Path = Path("/app/pipeline_state/cache")
    image_dir: Path = Path("dataset/arg_max/images")
    
    # Data source mappings
    url_map: Mapping[str, str] = field(default_factory=lambda: {
        "allrecipes.parquet": "/app/data/allrecipes.parquet",
        "ground_truth_sample.csv": "/app/data/ground_truth_sample.csv",
    })
    
    # Vectorizer configuration
    vec_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        min_df=2,
        ngram_range=(1, 3),
        max_features=50000,
        sublinear_tf=True
    ))
    
    # Memory thresholds
    memory_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 16.0,      # GB for high memory mode
        'medium': 8.0,     # GB for medium memory mode
        'critical': 0.9,   # 90% memory usage is critical
        'warning': 0.85,   # 85% memory usage is warning
        'memmap': 10000,   # Use memmap for datasets larger than this
    })
    
    # Network configuration
    network_config: Dict[str, Any] = field(default_factory=lambda: {
        'connectivity_test_urls': [
            "https://www.google.com",
            "https://api.github.com"
        ],
        'download_timeout': 30,
        'retry_delays': [0.5, 1, 2],  # Retry delays in seconds
        'max_download_workers': 16,
    })
    
    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        'min_images_for_training': 10,
        'sanity_check_fraction': 0.01,
        'checkpoint_save_frequency': 10,  # Save checkpoint every N batches
    })
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        # Create missing directories
        for field_name, value in self.__dict__.items():
            if isinstance(value, Path) and field_name not in ['url_map']:
                if not value.exists() and not str(value).startswith('http'):
                    try:
                        value.mkdir(parents=True, exist_ok=True)
                        logging.getLogger("PIPE").info(f"Created missing directory: {value}")
                    except Exception as e:
                        logging.getLogger("PIPE").warning(f"Could not create {value}: {e}")


# Create the configuration instance
CFG = Config()