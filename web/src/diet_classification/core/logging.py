#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized logging configuration for diet classification pipeline.

"""

import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# Module-level state
_logger = None
_initialized = False
_file_lock = threading.Lock()


class DirectWriteHandler(logging.Handler):
    """Handler that writes directly to file with no buffering"""

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        # Write a separator for new runs (but don't delete existing content)
        self._write_direct("\n" + "="*80 + "\n")
        self._write_direct(
            f"NEW RUN STARTED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._write_direct(f"Process ID: {os.getpid()}\n")
        self._write_direct("="*80 + "\n")

    def _write_direct(self, msg):
        """Write directly to file with no buffering"""
        with _file_lock:
            # Always append, never truncate
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(msg)
                f.flush()
                # Force OS-level flush
                try:
                    os.fsync(f.fileno())
                except:
                    pass  # Some filesystems don't support fsync

    def emit(self, record):
        try:
            msg = self.format(record)
            self._write_direct(msg + '\n')
        except Exception:
            self.handleError(record)


def log_exception_hook(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = get_logger()
    logger.error("Uncaught exception:", exc_info=(
        exc_type, exc_value, exc_traceback))


def setup_logging(log_dir: Path = None):
    """
    Initialize the logging system once.
    
    Args:
        log_dir: Directory for log files. If None, uses config default.
    
    Returns:
        Logger instance
    """
    global _logger, _initialized
    
    if _initialized:
        return _logger
    
    # Import here to avoid circular dependency
    from ..config import CFG
    
    if log_dir is None:
        log_dir = CFG.logs_dir
    
    # Make sure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define log file path
    log_file = log_dir / "diet_classifiers.py.log"
    
    # Get logger
    _logger = logging.getLogger("PIPE")
    
    # Only configure if no handlers exist
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)

        # Define formatter
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)s │ %(message)s", datefmt="%H:%M:%S")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

        # Direct write file handler
        direct_handler = DirectWriteHandler(str(log_file))
        direct_handler.setFormatter(formatter)
        _logger.addHandler(direct_handler)

        # Test logging
        _logger.info("Logging system initialized successfully")
    
    # Set exception hook
    sys.excepthook = log_exception_hook
    
    _initialized = True
    return _logger


def get_logger():
    """
    Get the singleton logger instance.
    
    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        setup_logging()
    return _logger