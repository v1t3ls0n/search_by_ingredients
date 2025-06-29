#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline checkpoint management for save/resume functionality.

Based on original lines 3231-3299 from diet_classifiers.py
"""

import sys
import pickle
from datetime import datetime
import psutil

from ..core import log
from ..config import CFG


def save_pipeline_state(stage: str, data: dict):
    """
    Save pipeline state for resume capability in persistent directory.

    Args:
        stage: Current pipeline stage identifier
        data: Dictionary of data to save
        
    Based on original lines 3231-3276
    """
    # Use the persistent state directory
    state_dir = CFG.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)

    state_path = state_dir / "pipeline_state.pkl"
    backup_path = state_dir / f"pipeline_state_{stage}.pkl"

    state = {
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
        'data': data,
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'artifacts_dir': str(CFG.artifacts_dir),  # Track where artifacts are
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

    try:
        # Save main state file
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

        # Save stage-specific backup
        with open(backup_path, 'wb') as f:
            pickle.dump(state, f)

        log.debug(f"   💾 Pipeline state saved at stage: {stage}")

        # Clean up old backups (keep only last 5)
        backups = sorted(state_dir.glob("pipeline_state_*.pkl"),
                         key=lambda p: p.stat().st_mtime)
        if len(backups) > 5:
            for old_backup in backups[:-5]:
                old_backup.unlink()
                log.debug(f"   🗑️  Removed old backup: {old_backup.name}")

    except Exception as e:
        log.error(f"   ❌ Failed to save pipeline state: {e}")


def load_pipeline_state():
    """
    Load the most recent pipeline state from persistent directory.

    Returns:
        Tuple of (stage, data) or (None, None) if no state found
        
    Based on original lines 3279-3299
    """
    state_path = CFG.state_dir / "pipeline_state.pkl"

    if not state_path.exists():
        return None, None

    try:
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        log.info(f"   📂 Loaded pipeline state from stage: {state['stage']}")
        log.info(f"   ├─ Saved at: {state['timestamp']}")
        log.info(f"   └─ Memory usage at save: {state['memory_usage']:.1f}%")

        return state['stage'], state['data']

    except Exception as e:
        log.error(f"   ❌ Failed to load pipeline state: {e}")
        return None, None