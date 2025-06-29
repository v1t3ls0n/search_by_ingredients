#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom exceptions for the diet classification pipeline.
"""


class DietClassifierError(Exception):
    """Base exception for all diet classifier errors."""
    pass


class DataLoadingError(DietClassifierError):
    """Raised when data loading fails."""
    pass


class ModelNotTrainedError(DietClassifierError):
    """Raised when trying to use models that haven't been trained."""
    pass


class FeatureExtractionError(DietClassifierError):
    """Raised when feature extraction fails."""
    pass


class ClassificationError(DietClassifierError):
    """Raised when classification fails."""
    pass


class ConfigurationError(DietClassifierError):
    """Raised when configuration is invalid."""
    pass