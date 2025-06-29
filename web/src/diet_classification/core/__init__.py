#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core module initialization - sets up shared state for the entire package.

This module ensures that logging and state management are initialized
exactly once when the package is imported.
"""

from .logging import setup_logging, get_logger
from .state import get_pipeline_state
from .exceptions import DietClassifierError

# Initialize logging system once
setup_logging()

# Get singleton logger instance for the package
log = get_logger()

# Log successful initialization
log.debug("Diet classification core module initialized")

# Export commonly used components
__all__ = [
    'log',
    'get_logger',
    'get_pipeline_state',
    'DietClassifierError',
]