#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image feature extraction module for diet classification.

This module implements:
- Robust image downloading with retry logic
- ResNet-50 feature extraction
- Embedding caching and management
- Quality filtering
"""

import os
import time
import json
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import joblib

# Optional imports with graceful fallback
try:
    import torch
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    models = None
    transforms = None
    Image = None

from ..core import log, get_pipeline_state
from ..config import CFG
from ..utils.memory import optimize_memory_usage, get_available_memory


def download_images(
    df: pd.DataFrame,
    output_dir: Path,
    max_workers: Optional[int] = None,
    force: bool = False,
    batch_size: int = 100
) -> List[int]:
    """
    Download images from URLs with robust retry and caching.
    
    Enhanced implementation with:
    - Embeddings-first approach (check if we already have embeddings)
    - Adaptive concurrency based on network conditions
    - Detailed progress tracking
    - Graceful error handling
    
    Args:
        df: DataFrame with 'photo_url' column
        output_dir: Directory to save images
        max_workers: Max concurrent downloads (auto-determined if None)
        force: Force redownload even if images exist
        batch_size: Process images in batches for memory efficiency
        
    Returns:
        List of successfully downloaded indices
    """
    if 'photo_url' not in df.columns:
        log.warning("No 'photo_url' column found in DataFrame")
        return []
    
    # Use config defaults
    if max_workers is None:
        max_workers = CFG.network_config['max_download_workers']
    
    # Check for existing embeddings first
    mode = output_dir.name
    embedding_candidates = [
        output_dir / "embeddings.npy",
        output_dir / f"embeddings_{mode}.npy",
        CFG.artifacts_dir / f"embeddings_{mode}.npy",
        CFG.artifacts_dir / f"embeddings_{mode}_backup.npy"
    ]
    
    if not force:
        for emb_path in embedding_candidates:
            if emb_path.exists():
                try:
                    embeddings = np.load(str(emb_path), mmap_mode='r')
                    if embeddings.shape[0] >= len(df) * 0.3:  # At least 30% coverage
                        log.info(f"✅ Found existing embeddings at {emb_path}")
                        log.info(f"   Shape: {embeddings.shape}")
                        log.info(f"   Skipping image downloads!")
                        return list(df.index[:embeddings.shape[0]])
                except Exception as e:
                    log.debug(f"Could not load embeddings from {emb_path}: {e}")
    
    # Proceed with downloads
    log.info(f"\n📥 DOWNLOADING IMAGES: {output_dir}")
    log.info(f"   Total URLs: {len(df):,}")
    log.info(f"   Max workers: {max_workers}")
    log.info(f"   Force redownload: {force}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze URLs and existing files
    download_queue = []
    existing_files = []
    invalid_urls = []
    
    for idx, row in df.iterrows():
        url = row.get('photo_url')
        
        if not url or not isinstance(url, str):
            invalid_urls.append(idx)
            continue
            
        url = str(url).strip()
        
        # Skip placeholder URLs
        if any(placeholder in url.lower() for placeholder in ['nophoto', 'nopic', 'nopicture']):
            invalid_urls.append(idx)
            continue
            
        # Check if file exists
        img_path = output_dir / f"{idx}.jpg"
        
        if img_path.exists() and not force:
            # Validate file size
            if img_path.stat().st_size > 100:
                existing_files.append(idx)
                continue
            else:
                # Remove corrupted file
                img_path.unlink()
        
        # Add to download queue
        download_queue.append((idx, url))
    
    log.info(f"\n📊 URL Analysis:")
    log.info(f"   ├─ Valid URLs: {len(download_queue):,}")
    log.info(f"   ├─ Existing files: {len(existing_files):,}")
    log.info(f"   └─ Invalid URLs: {len(invalid_urls):,}")
    
    if not download_queue:
        log.info("✅ All files already exist")
        return existing_files
    
    # Download in batches
    successful_downloads = existing_files.copy()
    failed_downloads = []
    
    # Process in batches for better memory management
    for batch_start in range(0, len(download_queue), batch_size):
        batch_end = min(batch_start + batch_size, len(download_queue))
        batch = download_queue[batch_start:batch_end]
        
        log.info(f"\n📦 Processing batch {batch_start//batch_size + 1}/{(len(download_queue) + batch_size - 1)//batch_size}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_single_image, idx, url, output_dir): (idx, url)
                for idx, url in batch
            }
            
            with tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"   Downloading batch",
                position=0,
                leave=False
            ) as pbar:
                for future in pbar:
                    idx, url = futures[future]
                    
                    try:
                        success, size = future.result()
                        
                        if success:
                            successful_downloads.append(idx)
                            pbar.set_postfix({
                                'Success': len(successful_downloads),
                                'Failed': len(failed_downloads)
                            })
                        else:
                            failed_downloads.append((idx, "Download failed"))
                            
                    except Exception as e:
                        failed_downloads.append((idx, str(e)))
        
        # Memory cleanup between batches
        optimize_memory_usage(f"After batch {batch_start//batch_size + 1}")
    
    # Summary
    log.info(f"\n📊 Download Summary:")
    log.info(f"   ├─ Successful: {len(successful_downloads):,}")
    log.info(f"   ├─ Failed: {len(failed_downloads):,}")
    log.info(f"   └─ Success rate: {len(successful_downloads)/len(df)*100:.1f}%")
    
    return sorted(successful_downloads)


def download_single_image(
    idx: int,
    url: str,
    output_dir: Path,
    max_retries: int = 3
) -> Tuple[bool, int]:
    """
    Download a single image with retry logic.
    
    Args:
        idx: Index for filename
        url: Image URL
        output_dir: Output directory
        max_retries: Maximum retry attempts
        
    Returns:
        Tuple of (success, file_size)
    """
    img_path = output_dir / f"{idx}.jpg"
    
    # Use config retry delays
    retry_delays = CFG.network_config['retry_delays']
    timeout = CFG.network_config['download_timeout']
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; DietClassifier/1.0)',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png']):
                raise ValueError(f"Invalid content type: {content_type}")
            
            # Download with size limit
            content = b''
            max_size = 20 * 1024 * 1024  # 20MB
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
                    if len(content) > max_size:
                        raise ValueError("File too large")
            
            # Validate minimum size
            if len(content) < 500:
                raise ValueError("File too small")
            
            # Save atomically
            temp_path = img_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            temp_path.rename(img_path)
            
            return True, len(content)
            
        except Exception as e:
            if attempt < max_retries - 1 and attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
            else:
                return False, 0
    
    return False, 0


def build_image_embeddings(
    df: pd.DataFrame,
    mode: str,
    force: bool = False,
    batch_size: Optional[int] = None,
    device: Optional[str] = None
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract ResNet-50 embeddings from images.
    
    Memory-efficient implementation with:
    - Adaptive batch sizing
    - Checkpoint/resume capability
    - GPU/CPU automatic selection
    - Quality filtering
    
    Args:
        df: DataFrame with image indices
        mode: 'silver' or 'gold'
        force: Force recomputation
        batch_size: Batch size for processing (auto if None)
        device: Device to use ('cuda', 'cpu', or None for auto)
        
    Returns:
        Tuple of (embeddings_array, valid_indices)
    """
    if not TORCH_AVAILABLE:
        log.error("PyTorch not available - cannot extract image embeddings")
        return np.zeros((0, 2048), dtype=np.float32), []
    
    log.info(f"\n🧠 EXTRACTING IMAGE EMBEDDINGS: {mode}")
    log.info(f"   Target images: {len(df):,}")
    log.info(f"   Force recompute: {force}")
    
    # Setup paths
    img_dir = CFG.image_dir / mode
    embed_path = img_dir / "embeddings.npy"
    metadata_path = img_dir / "embedding_metadata.json"
    checkpoint_path = img_dir / "embedding_checkpoint.npz"
    
    # Check for existing embeddings
    if not force and embed_path.exists():
        try:
            embeddings = np.load(embed_path)
            if embeddings.shape[0] == len(df):
                log.info(f"✅ Using cached embeddings from {embed_path}")
                return embeddings, list(df.index)
        except Exception as e:
            log.warning(f"Could not load cached embeddings: {e}")
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    log.info(f"   Device: {device}")
    
    # Adaptive batch size
    if batch_size is None:
        available_memory = get_available_memory()
        if device.type == 'cuda':
            # GPU memory is more limited
            batch_size = min(32, max(1, int(available_memory * 2)))
        else:
            # CPU can handle larger batches
            batch_size = min(64, max(1, int(available_memory * 5)))
    
    log.info(f"   Batch size: {batch_size}")
    
    # Load model
    log.info("   Loading ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # Remove classification layer
    model.eval()
    model.to(device)
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Check for checkpoint
    start_idx = 0
    embeddings_list = []
    valid_indices = []
    
    if checkpoint_path.exists() and not force:
        try:
            checkpoint = np.load(checkpoint_path, allow_pickle=True)
            start_idx = int(checkpoint['last_idx']) + 1
            embeddings_list = checkpoint['embeddings'].tolist()
            valid_indices = checkpoint['indices'].tolist()
            log.info(f"   Resuming from checkpoint: {start_idx}/{len(df)}")
        except Exception as e:
            log.warning(f"Could not load checkpoint: {e}")
    
    # Process images
    with tqdm(
        range(start_idx, len(df)),
        initial=start_idx,
        total=len(df),
        desc="   Extracting features"
    ) as pbar:
        
        batch_images = []
        batch_indices = []
        
        for i in pbar:
            idx = df.index[i]
            img_path = img_dir / f"{idx}.jpg"
            
            if not img_path.exists():
                continue
            
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                
                batch_images.append(img_tensor)
                batch_indices.append(idx)
                
                # Process batch
                if len(batch_images) >= batch_size or i == len(df) - 1:
                    batch_tensor = torch.cat(batch_images).to(device)
                    
                    with torch.no_grad():
                        batch_embeddings = model(batch_tensor).cpu().numpy()
                    
                    embeddings_list.extend(batch_embeddings)
                    valid_indices.extend(batch_indices)
                    
                    # Clear batch
                    batch_images = []
                    batch_indices = []
                    
                    # Update progress
                    pbar.set_postfix({
                        'Valid': len(valid_indices),
                        'Batch': f"{len(batch_embeddings)}"
                    })
                    
            except Exception as e:
                log.debug(f"Failed to process image {idx}: {e}")
                continue
            
            # Periodic checkpoint
            if i > 0 and i % 100 == 0:
                save_embedding_checkpoint(
                    checkpoint_path,
                    i,
                    embeddings_list,
                    valid_indices
                )
    
    # Convert to array
    if embeddings_list:
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Quality filtering
        embeddings, valid_indices = filter_low_quality_embeddings(
            embeddings,
            valid_indices
        )
        
        # Save embeddings
        np.save(embed_path, embeddings)
        
        # Save metadata
        metadata = {
            'creation_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'mode': mode,
            'total_images': len(df),
            'valid_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'model': 'ResNet-50',
            'device': str(device)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"\n✅ Embeddings extracted successfully")
        log.info(f"   ├─ Shape: {embeddings.shape}")
        log.info(f"   └─ Saved to: {embed_path}")
        
        # Cleanup checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        return embeddings, valid_indices
    else:
        log.warning("No valid embeddings extracted")
        return np.zeros((0, 2048), dtype=np.float32), []


def filter_low_quality_embeddings(
    embeddings: np.ndarray,
    indices: List[int],
    variance_percentile: int = 10,
    min_retention: float = 0.5
) -> Tuple[np.ndarray, List[int]]:
    """
    Filter out low-quality image embeddings.
    
    Removes embeddings that likely correspond to:
    - Blank or corrupted images (very low variance)
    - Generic placeholder images (too similar to mean)
    
    Args:
        embeddings: Embedding array
        indices: Corresponding indices
        variance_percentile: Percentile threshold for variance
        min_retention: Minimum fraction of embeddings to keep
        
    Returns:
        Filtered (embeddings, indices)
    """
    if len(embeddings) == 0:
        return embeddings, indices
    
    # Calculate statistics
    variances = np.var(embeddings, axis=1)
    means = np.mean(embeddings, axis=1)
    
    # Define thresholds
    var_threshold = np.percentile(variances, variance_percentile)
    mean_threshold = np.percentile(means, 90)
    
    # Create quality mask
    quality_mask = (variances > var_threshold) & (means < mean_threshold)
    
    # Ensure minimum retention
    if quality_mask.sum() < len(embeddings) * min_retention:
        # Keep top embeddings by variance
        n_keep = int(len(embeddings) * min_retention)
        top_indices = np.argsort(variances)[-n_keep:]
        quality_mask = np.zeros(len(embeddings), dtype=bool)
        quality_mask[top_indices] = True
    
    # Apply filter
    filtered_embeddings = embeddings[quality_mask]
    filtered_indices = [indices[i] for i in range(len(indices)) if quality_mask[i]]
    
    log.info(f"   Quality filtering: {len(filtered_indices)}/{len(indices)} kept")
    
    return filtered_embeddings, filtered_indices


def save_embedding_checkpoint(
    path: Path,
    last_idx: int,
    embeddings: List[np.ndarray],
    indices: List[int]
):
    """Save checkpoint for resume capability."""
    checkpoint = {
        'last_idx': last_idx,
        'embeddings': np.array(embeddings),
        'indices': np.array(indices),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    np.savez_compressed(path, **checkpoint)


def load_image_embeddings(mode: str) -> Optional[np.ndarray]:
    """
    Load pre-computed image embeddings.
    
    Args:
        mode: 'silver' or 'gold'
        
    Returns:
        Embeddings array or None if not found
    """
    paths = [
        CFG.image_dir / mode / "embeddings.npy",
        CFG.artifacts_dir / f"embeddings_{mode}.npy",
        CFG.artifacts_dir / f"embeddings_{mode}_backup.npy"
    ]
    
    for path in paths:
        if path.exists():
            try:
                embeddings = np.load(path)
                log.info(f"Loaded embeddings from {path}: shape {embeddings.shape}")
                return embeddings
            except Exception as e:
                log.warning(f"Could not load embeddings from {path}: {e}")
    
    return None


def validate_image_alignment(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    indices: List[int]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Ensure DataFrame and embeddings are properly aligned.
    
    Args:
        df: DataFrame with image data
        embeddings: Embedding array
        indices: Valid indices from embedding extraction
        
    Returns:
        Aligned (DataFrame, embeddings)
    """
    if len(df) != len(embeddings):
        log.warning(f"Alignment mismatch: DataFrame has {len(df)} rows, embeddings has {len(embeddings)}")
        
        # Align to valid indices
        if indices:
            df_aligned = df.loc[indices].copy()
            
            # Ensure embeddings match
            if len(df_aligned) != len(embeddings):
                min_len = min(len(df_aligned), len(embeddings))
                df_aligned = df_aligned.iloc[:min_len]
                embeddings = embeddings[:min_len]
            
            return df_aligned, embeddings
        else:
            # Truncate to smaller size
            min_len = min(len(df), len(embeddings))
            return df.iloc[:min_len].copy(), embeddings[:min_len]
    
    return df, embeddings