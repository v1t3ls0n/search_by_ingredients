#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory management utilities for the diet classification pipeline.

Based on original lines 634-902 from diet_classifiers.py
"""

import gc
import os
import platform
import psutil

# Import shared components
from ..core import log, get_pipeline_state
from ..config import CFG


def get_available_memory(safety_factor=0.9):
    """
    Get available memory accounting for Docker container limits.
    Enhanced version that properly detects container memory constraints.
    
    Based on original lines 634-703
    """
    mem = psutil.virtual_memory()
    total_memory = mem.total

    # Check Docker cgroup limits
    cgroup_files = [
        '/sys/fs/cgroup/memory/memory.limit_in_bytes',  # cgroup v1
        '/sys/fs/cgroup/memory.max',  # cgroup v2
    ]

    for cgroup_file in cgroup_files:
        if os.path.exists(cgroup_file):
            try:
                with open(cgroup_file, 'r') as f:
                    limit = f.read().strip()
                    if limit != 'max' and limit.isdigit():
                        cgroup_limit = int(limit)
                        # Use the smaller of cgroup limit or system memory
                        if cgroup_limit < total_memory * 2:  # sanity check
                            total_memory = min(total_memory, cgroup_limit)
                            log.info(
                                f"Container memory limited by cgroup: {total_memory / (1024**3):.1f} GB")
                            break
            except Exception as e:
                log.warning(f"Could not read cgroup file {cgroup_file}: {e}")

    # Add check for Docker Desktop on Mac/Windows
    if platform.system() in ['Darwin', 'Windows']:
        # Try to read from Docker's memory limit file
        docker_limit_file = '/.dockerenv'
        if os.path.exists(docker_limit_file):
            log.warning(
                "Running in Docker on Mac/Windows - memory limits may not be accurately detected")
            # Conservative estimate for Docker Desktop
            estimated_limit_gb = min(
                8.0, total_memory / (1024**3))  # Default to 8GB max
            log.info(
                f"Using conservative Docker Desktop limit: {estimated_limit_gb:.1f} GB")
            return estimated_limit_gb * safety_factor

    # Return usable memory in GB
    usable_gb = (total_memory * safety_factor) / (1024**3)

    log.info(
        f"System/Container total memory: {total_memory / (1024**3):.1f} GB")
    log.info(
        f"Safe memory limit ({safety_factor*100:.0f}%): {usable_gb:.1f} GB")

    return usable_gb


def optimize_memory_usage(stage_name=""):
    """
    Optimize memory usage during training with enhanced Docker support.
    
    Based on original lines 704-795
    """
    # Get memory before cleanup
    try:
        memory_before = psutil.virtual_memory()
        memory_before_used = memory_before.used
        memory_before_percent = memory_before.percent

        # Log current memory state with container awareness
        available_memory_gb = get_available_memory(
            safety_factor=1.0)  # Get total available
        log.info(f"   🧹 {stage_name}: Memory optimization")
        log.info(
            f"      ├─ Container/System memory: {available_memory_gb:.1f} GB total")
        log.info(
            f"      ├─ Currently used: {memory_before_percent:.1f}% ({memory_before_used / (1024**2):.0f} MB)")

    except Exception as e:
        log.error(f"Failed to get initial memory stats: {e}")
        return "error"

    # Force garbage collection multiple times
    collected_total = 0
    for i in range(3):
        try:
            collected = gc.collect()
            collected_total += collected
        except Exception as e:
            log.debug(f"Garbage collection pass {i+1} failed: {e}")

    # Clear GPU cache if available
    gpu_freed = 0
    try:
        import torch
        if torch.cuda.is_available():
            try:
                gpu_before = torch.cuda.memory_allocated() / (1024**2)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure operations complete
                gpu_after = torch.cuda.memory_allocated() / (1024**2)
                gpu_freed = max(0, gpu_before - gpu_after)
            except Exception as e:
                log.debug(f"GPU memory cleanup failed: {e}")
    except ImportError:
        pass  # PyTorch not available

    # Get memory after cleanup
    try:
        memory_after = psutil.virtual_memory()
        memory_freed_bytes = max(0, memory_before_used - memory_after.used)
        memory_freed_mb = memory_freed_bytes / (1024**2)

    except Exception as e:
        log.error(f"Failed to get final memory stats: {e}")
        return "error"

    # Log results
    if memory_freed_mb > 1.0 or collected_total > 0 or gpu_freed > 1.0:
        log.info(f"      ├─ Freed: {memory_freed_mb:.1f} MB RAM")
        if collected_total > 0:
            log.info(f"      ├─ Objects collected: {collected_total}")
        if gpu_freed > 1.0:
            log.info(f"      ├─ GPU freed: {gpu_freed:.1f} MB")

    # Check final status
    if memory_after.percent > 90:
        log.error(
            f"      ❌ CRITICAL memory usage: {memory_after.percent:.1f}%")
        # Try emergency cleanup
        handle_memory_crisis()
        return "critical"
    elif memory_after.percent > 85:
        log.warning(
            f"      ⚠️  High memory usage: {memory_after.percent:.1f}%")
        return "high"
    elif memory_after.percent > 70:
        log.info(
            f"      ⚠️  Moderate memory usage: {memory_after.percent:.1f}%")
        return "moderate"
    else:
        log.info(f"      ✅ Memory usage normal: {memory_after.percent:.1f}%")
        return "normal"


def handle_memory_crisis():
    """
    Emergency memory cleanup when usage is critical.

    Enhanced with adaptive model switching and disk-based processing.
    
    Based on original lines 796-902
    """
    log.warning("🚨 MEMORY CRISIS - Applying emergency cleanup")
    
    # Get pipeline state
    pipeline_state = get_pipeline_state()

    try:
        initial_memory = psutil.virtual_memory()
        initial_percent = initial_memory.percent
        log.info(f"   ├─ Initial memory: {initial_percent:.1f}%")

        # Step 1: Multiple aggressive garbage collection passes
        total_collected = 0
        for i in range(5):
            try:
                collected = gc.collect()
                total_collected += collected
                if collected > 0:
                    log.info(
                        f"   ├─ GC pass {i+1}: {collected} objects collected")
            except Exception as e:
                log.debug(f"GC pass {i+1} failed: {e}")

        # Step 2: Clear all GPU memory
        gpu_freed = 0
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    gpu_before = torch.cuda.memory_allocated() / (1024**2)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gpu_after = torch.cuda.memory_allocated() / (1024**2)
                    gpu_freed = max(0, gpu_before - gpu_after)
                    log.info(f"   ├─ GPU memory freed: {gpu_freed:.1f} MB")
                except Exception as e:
                    log.debug(f"   ├─ GPU cleanup failed: {e}")
        except ImportError:
            pass

        # Step 3: Clear Python internal caches
        try:
            import importlib
            if hasattr(importlib, 'invalidate_caches'):
                importlib.invalidate_caches()
        except Exception as e:
            log.debug(f"   ├─ Cache cleanup failed: {e}")

        # Step 4: Force memory compaction
        try:
            gc.set_debug(0)
            gc.collect()
        except Exception:
            pass

        # Step 5: Check final memory
        final_memory = psutil.virtual_memory()
        final_percent = final_memory.percent
        memory_freed_mb = (initial_memory.used - final_memory.used) / (1024**2)

        log.info(f"   ├─ Objects collected: {total_collected}")
        log.info(f"   ├─ Memory freed: {memory_freed_mb:.1f} MB")
        log.info(f"   └─ Final memory usage: {final_percent:.1f}%")

        # Adaptive strategy based on final memory
        if final_percent > CFG.memory_thresholds['critical'] * 100:
            log.error(f"   ❌ Still critical! Switching to emergency mode")

            # Force minimal models
            os.environ['FORCE_MINIMAL_MODELS'] = '1'

            # Switch to disk-based processing
            os.environ['USE_DISK_CACHE'] = '1'

            # Reduce batch sizes
            os.environ['BATCH_SIZE_MULTIPLIER'] = '0.25'

            # Update pipeline state
            pipeline_state.memory_mode = 'critical'

            log.info(f"   🚨 Emergency measures activated:")
            log.info(f"      ├─ Minimal models only")
            log.info(f"      ├─ Disk-based caching enabled")
            log.info(f"      └─ Batch sizes reduced to 25%")

        return final_percent

    except Exception as e:
        log.error(f"Memory crisis handling failed: {e}")
        return 90.0  # Assume high usage if we can't measure