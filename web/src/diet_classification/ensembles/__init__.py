#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble methods for diet classification.

This module provides various ensemble techniques including:
- Voting ensembles (hard/soft)
- Blending methods
- Stacking
- Cross-domain ensembles
- Ensemble optimization
"""

# Base classes
from .base import (
    BaseEnsemble,
    EnsembleWrapper,
    create_ensemble,
    evaluate_ensemble_diversity
)

# Voting ensembles
from .voting import (
    create_voting_ensemble,
    dynamic_weighted_voting,
    threshold_voting_ensemble,
    ranked_voting_ensemble,
    calibrated_voting_ensemble,
    hierarchical_voting_ensemble
)

# Blending methods
from .blending import (
    best_two_domains,
    linear_blending,
    ranked_average_blending,
    stacking_blend,
    dynamic_blending,
    trimmed_mean_blending,
    create_simple_average_ensemble
)

# Optimization methods
from .optimization import (
    top_n_ensemble,
    best_ensemble,
    greedy_ensemble_selection,
    ensemble_pruning
)

__all__ = [
    # Base
    'BaseEnsemble',
    'EnsembleWrapper',
    'create_ensemble',
    'evaluate_ensemble_diversity',
    
    # Voting
    'create_voting_ensemble',
    'dynamic_weighted_voting',
    'threshold_voting_ensemble',
    'ranked_voting_ensemble',
    'calibrated_voting_ensemble',
    'hierarchical_voting_ensemble',
    
    # Blending
    'best_two_domains',
    'linear_blending',
    'ranked_average_blending',
    'stacking_blend',
    'dynamic_blending',
    'trimmed_mean_blending',
    'create_simple_average_ensemble',
    
    # Optimization
    'top_n_ensemble',
    'best_ensemble',
    'greedy_ensemble_selection',
    'ensemble_pruning'
]