#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Mapping
import logging
from dataclasses import dataclass, field
@dataclass(frozen=True)
class Config:
    """
    Immutable configuration container for the pipeline with validation.
    """
    pretrained_models_dir: Path = Path("/app/pretrained_models")
    artifacts_dir: Path = Path("/app/artifacts")
    logs_dir: Path = Path("/app/artifacts/logs")
    data_dir: Path = Path("/app/data")
    usda_dir: Path = Path("/app/data/usda")
    state_dir: Path = Path("/app/pipeline_state")
    checkpoints_dir: Path = Path("/app/pipeline_state/checkpoints")
    cache_dir: Path = Path("/app/pipeline_state/cache")
    url_map: Mapping[str, str] = field(default_factory=lambda: {
        "allrecipes.parquet": "/app/data/allrecipes.parquet",
        "ground_truth_sample.csv": "/app/data/ground_truth_sample.csv",
    })
    vec_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        min_df=2, ngram_range=(1, 3), max_features=50000, sublinear_tf=True))
    image_dir: Path = Path("dataset/arg_max/images")

    # NEW: Configurable thresholds
    memory_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'high': 16.0,      # GB for high memory mode
        'medium': 8.0,     # GB for medium memory mode
        'critical': 0.9,   # 90% memory usage is critical
        'warning': 0.85,   # 85% memory usage is warning
        'memmap': 10000,   # Use memmap for datasets larger than this
    })

    # NEW: Network configuration
    network_config: Dict[str, Any] = field(default_factory=lambda: {
        'connectivity_test_urls': [
            "https://www.google.com",
            "https://api.github.com"
        ],
        'download_timeout': 30,
        'retry_delays': [0.5, 1, 2],  # Faster retry delays
        'max_download_workers': 16,
    })

    # NEW: Model configuration
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        'min_images_for_training': 10,
        'sanity_check_fraction': 0.01,
        'checkpoint_save_frequency': 10,  # Save checkpoint every N batches
    })

    def __post_init__(self):
        """Validate configuration on initialization."""
        # Create missing directories
        for field_name, value in self.__dictz__.items():
            if isinstance(value, Path) and field_name != 'url_map':
                if not value.exists() and not str(value).startswith('http'):
                    try:
                        value.mkdir(parents=True, exist_ok=True)
                        logging.getLogger("PIPE").info(
                            f"Created missing directory: {value}")
                    except Exception as e:
                        logging.getLogger("PIPE").warning(
                            f"Could not create {value}: {e}")


# Create the new config instance
CFG = Config()