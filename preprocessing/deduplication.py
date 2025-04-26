#!/usr/bin/env python3
"""
Sign Language Datasets Deduplication Module

This module handles duplicate detection and removal in sign language image datasets:
- Perceptual hash-based duplicate detection with adaptive thresholding
- Removal of duplicate images while preserving class distributions
- Comparative analysis between original and deduplicated datasets

The module creates deduplicated datasets ready for further processing.
"""
#Author: MatheusHRV

import os
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
import glob
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import random
import cv2
from collections import defaultdict
import warnings
import multiprocessing
from functools import partial
import concurrent.futures
import time
warnings.filterwarnings('ignore')

# Import from other modules
from preprocessing import analyze_image_dataset_metadata, OPTIMAL_WORKERS
from eda import perform_image_eda
from visualization import plot_class_distribution, plot_duplicate_summary

# Set constants
HASH_SIZE = 16  # Hash size for perceptual hashing (higher = more sensitive)
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']

# Create necessary directories
os.makedirs('deduplicated_datasets', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('excel', exist_ok=True)

# Thread-based file reader (faster than sequential I/O)
class AsyncImageLoader:
    def __init__(self, file_list, max_queue_size=100):
        self.file_list = file_list
        self.queue = multiprocessing.Queue(maxsize=max_queue_size)
        self.stopped = False
        
    def start(self):
        multiprocessing.Process(target=self.fill_queue, daemon=True).start()
        return self
        
    def fill_queue(self):
        for file_path in self.file_list:
            if self.stopped:
                break
            try:
                img = Image.open(file_path).convert('RGB')
                self.queue.put((file_path, img))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        # Add None to signal the end
        self.queue.put((None, None))
        
    def get(self):
        return self.queue.get()
        
    def stop(self):
        self.stopped = True

# Function to compute perceptual hash for an image
def compute_phash(image_path):
    """
    Compute perceptual hash for an image
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        tuple: (image_path, hash)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        phash = imagehash.phash(img, hash_size=HASH_SIZE)
        return (image_path, phash)
    except Exception as e:
        print(f"Error computing hash for {image_path}: {e}")
        return (image_path, None)

# Function to compute perceptual hashes in parallel
def compute_phashes_parallel(image_paths, num_workers=None):
    """
    Compute perceptual hashes for a list of images in parallel
    
    Args:
        image_paths (list): List of image paths
        num_workers (int): Number of worker processes
        
    Returns:
        dict: Dictionary with image paths as keys and hashes as values
    """
    if num_workers is None:
        num_workers = min(OPTIMAL_WORKERS, len(image_paths))
        
    print(f"Computing perceptual hashes for {len(image_paths)} images using {num_workers} workers...")
    image_hashes = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(compute_phash, image_paths), total=len(image_paths), desc="Computing hashes"))
        
    for result in results:
        image_path, phash = result
        if phash is not None:
            image_hashes[image_path] = phash
    
    print(f"  Computed {len(image_hashes)} valid hashes")
    return image_hashes

# Function to find duplicates based on perceptual hashes
def find_duplicates(image_hashes, threshold=4):
    """
    Find duplicates based on perceptual hashes
    
    Args:
        image_hashes (dict): Dictionary with image paths as keys and hashes as values
        threshold (int): Hamming distance threshold for considering images as duplicates
        
    Returns:
        dict: Dictionary with groups of duplicate images
    """
    print(f"Finding duplicates with threshold {threshold}...")
    
    # Organize hashes by value for faster comparison
    hash_dict = defaultdict(list)
    for img_path, phash in image_hashes.items():
        hash_dict[phash].append(img_path)
    
    # Find exact hash duplicates
    exact_duplicates = {hash_val: paths for hash_val, paths in hash_dict.items() if len(paths) > 1}
    
    # Find near duplicates
    duplicate_groups = list(exact_duplicates.values())
    print(f"  Found {len(duplicate_groups)} groups of exact duplicates")
    
    # If threshold > 0, find near duplicates
    if threshold > 0:
        print("  Looking for near duplicates...")
        hash_list = list(hash_dict.keys())
        
        # Compare each hash with others
        near_duplicate_groups = []
        
        # Use a more efficient approach for large datasets
        if len(hash_list) > 1000:
            # Use approximate nearest neighbors approach
            from collections import defaultdict
            
            # Simplified implementation for illustration
            buckets = defaultdict(list)
            
            # Create buckets of similar hashes
            for i, phash in enumerate(tqdm(hash_list, desc="Creating hash buckets")):
                # Use the first bits of the hash as a bucket key
                bucket_key = str(phash)[:4]  # Use first 4 characters as bucket key
                buckets[bucket_key].append((i, phash))
            
            # Check for near duplicates within each bucket
            processed = set()
            for bucket in tqdm(buckets.values(), desc="Checking buckets for near duplicates"):
                for i, (idx1, hash1) in enumerate(bucket):
                    if idx1 in processed:
                        continue
                        
                    group = []
                    for idx2, hash2 in bucket[i+1:]:
                        if idx2 in processed:
                            continue
                            
                        # Check if hashes are near duplicates
                        if hash1 - hash2 <= threshold:
                            if not group:  # First match, add the original hash
                                group = hash_dict[hash_list[idx1]].copy()
                                processed.add(idx1)
                            
                            # Add near duplicate
                            group.extend(hash_dict[hash_list[idx2]])
                            processed.add(idx2)
                    
                    if group:
                        near_duplicate_groups.append(group)
        else:
            # For smaller datasets, use pairwise comparison
            for i in tqdm(range(len(hash_list)), desc="Finding near duplicates"):
                if i % 100 == 0:
                    print(f"  Checked {i}/{len(hash_list)} hashes")
                    
                hash1 = hash_list[i]
                
                # Skip if already in a group
                if any(any(img_path in hash_dict[hash1] for img_path in group) for group in near_duplicate_groups):
                    continue
                
                group = []
                for j in range(i+1, len(hash_list)):
                    hash2 = hash_list[j]
                    
                    # Check if hashes are near duplicates
                    if hash1 - hash2 <= threshold:
                        if not group:  # First match, add the original hash
                            group = hash_dict[hash1].copy()
                        
                        # Add near duplicate
                        group.extend(hash_dict[hash2])
                
                if group:
                    near_duplicate_groups.append(group)
        
        # Add near duplicate groups to the result
        duplicate_groups.extend(near_duplicate_groups)
        print(f"  Found {len(near_duplicate_groups)} groups of near duplicates")
    
    print(f"  Found {len(duplicate_groups)} groups of duplicates in total")
    return duplicate_groups

# Function to create deduplicated datasets
def create_deduplicated_dataset(dataset_path, duplicate_groups, output_path):
    """
    Create a deduplicated dataset by selecting one image from each duplicate group
    
    Args:
        dataset_path (str): Path to the original dataset
        duplicate_groups (list): List of groups of duplicate images
        output_path (str): Path to save the deduplicated dataset
        
    Returns:
        tuple: (dict with stats, list of removed images, list of kept images)
    """
    print(f"\nCreating deduplicated dataset at {output_path}...")
    
    # Make sure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Get all images in the dataset
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    
    print(f"  Found {len(all_images)} images in the original dataset")
    
    # Create a set of all duplicate images
    all_duplicates = set()
    for group in duplicate_groups:
        all_duplicates.update(group)
    
    # Find which images need to be kept
    images_to_keep = []
    images_to_remove = []
    
    # Create a mapping of duplicates to their groups
    duplicate_to_group = {}
    for i, group in enumerate(duplicate_groups):
        for img_path in group:
            duplicate_to_group[img_path] = i
    
    # Process each duplicate group
    for group in duplicate_groups:
        # Select one image to keep from each group
        # Prefer to keep images with longer paths (usually contain more metadata)
        selected = max(group, key=lambda x: len(x))
        
        # Keep track of which images to keep and remove
        images_to_keep.append(selected)
        for img_path in group:
            if img_path != selected:
                images_to_remove.append(img_path)
    
    # Keep all non-duplicate images
    for img_path in all_images:
        if img_path not in all_duplicates:
            images_to_keep.append(img_path)
    
    # Copy kept images to the output directory
    print(f"  Copying {len(images_to_keep)} images to the deduplicated dataset...")
    
    # Get class distribution in original dataset
    original_class_counts = defaultdict(int)
    for img_path in all_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        original_class_counts[class_name] += 1
    
    # Count kept images by class
    kept_class_counts = defaultdict(int)
    for img_path in images_to_keep:
        class_name = os.path.basename(os.path.dirname(img_path))
        kept_class_counts[class_name] += 1
    
    # Create stats dictionary
    stats = {
        'total_original_images': len(all_images),
        'total_duplicate_groups': len(duplicate_groups),
        'total_duplicate_images': len(all_duplicates),
        'unique_duplicates': len(set(all_duplicates)),
        'images_to_remove': len(images_to_remove),
        'images_to_keep': len(images_to_keep),
        'removed_percentage': len(images_to_remove) / len(all_images) * 100 if all_images else 0,
        'original_class_counts': dict(original_class_counts),
        'kept_class_counts': dict(kept_class_counts),
    }
    
    # Copy kept images to output directory
    for img_path in tqdm(images_to_keep, desc="Copying kept images"):
        # Determine relative path
        rel_path = os.path.relpath(img_path, start=dataset_path)
        dst_path = os.path.join(output_path, rel_path)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(img_path, dst_path)
    
    print(f"  Deduplicated dataset created at {output_path}")
    print(f"  Removed {len(images_to_remove)} duplicate images ({stats['removed_percentage']:.2f}%)")
    
    return stats, images_to_remove, images_to_keep

# Function to find the optimal threshold for duplicate detection
def find_optimal_threshold(image_hashes, sample_size=100, max_threshold=16):
    """
    Find the optimal threshold for duplicate detection
    
    Args:
        image_hashes (dict): Dictionary with image paths as keys and hashes as values
        sample_size (int): Sample size for testing thresholds
        max_threshold (int): Maximum threshold to test
        
    Returns:
        int: Optimal threshold
    """
    print(f"Finding optimal threshold using {sample_size} sample images...")
    
    # Sample some images
    sample_paths = random.sample(list(image_hashes.keys()), min(sample_size, len(image_hashes)))
    
    # Calculate distances between pairs
    distances = []
    for i in range(len(sample_paths)):
        hash1 = image_hashes[sample_paths[i]]
        for j in range(i+1, len(sample_paths)):
            hash2 = image_hashes[sample_paths[j]]
            distance = hash1 - hash2
            distances.append(distance)
    
    # Find the elbow point in the sorted distances
    sorted_distances = sorted(distances)
    optimal_idx = 0
    max_curvature = 0
    
    for i in range(1, len(sorted_distances) - 1):
        x1, y1 = i-1, sorted_distances[i-1]
        x2, y2 = i, sorted_distances[i]
        x3, y3 = i+1, sorted_distances[i+1]
        
        # Calculate angle
        dx1, dy1 = x2-x1, y2-y1
        dx2, dy2 = x3-x2, y3-y2
        
        # Normalize
        len1 = (dx1**2 + dy1**2)**0.5
        len2 = (dx2**2 + dy2**2)**0.5
        
        dx1, dy1 = dx1/len1, dy1/len1
        dx2, dy2 = dx2/len2, dy2/len2
        
        # Dot product of normalized vectors
        dot = dx1*dx2 + dy1*dy2
        angle = abs(1 - dot)  # 0 = straight line, 2 = complete reversal
        
        if angle > max_curvature:
            max_curvature = angle
            optimal_idx = i
    
    optimal_threshold = min(sorted_distances[optimal_idx], max_threshold)
    print(f"  Optimal threshold determined to be: {optimal_threshold}")
    
    return optimal_threshold

# Main deduplication function
def deduplicate_dataset(dataset_path, dataset_name, threshold=None):
    """
    Deduplicate a dataset
    
    Args:
        dataset_path (str): Path to the dataset
        dataset_name (str): Name of the dataset
        threshold (int): Threshold for duplicate detection (None for automatic)
        
    Returns:
        tuple: (output path, stats, removed images, kept images)
    """
    print(f"\nDeduplicating dataset: {dataset_name}")
    output_path = os.path.join("deduplicated_datasets", dataset_name)
    
    # Get all images in the dataset
    all_images = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    
    if not all_images:
        print(f"  No images found in {dataset_path}!")
        return output_path, {
            'total_original_images': 0,
            'total_duplicate_groups': 0,
            'total_duplicate_images': 0,
            'unique_duplicates': 0,
            'images_to_remove': 0,
            'images_to_keep': 0,
            'removed_percentage': 0,
            'original_class_counts': {},
            'kept_class_counts': {},
        }, [], []
    
    print(f"  Found {len(all_images)} images")
    
    # Compute perceptual hashes
    image_hashes = compute_phashes_parallel(all_images)
    
    # Determine threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(image_hashes)
    
    # Find duplicates
    duplicate_groups = find_duplicates(image_hashes, threshold)
    
    # Create deduplicated dataset
    stats, removed_images, kept_images = create_deduplicated_dataset(
        dataset_path, duplicate_groups, output_path)
    
    return output_path, stats, removed_images, kept_images

def main():
    """Main function to run the deduplication pipeline"""
    print("Starting ASL and Libras Datasets Deduplication Pipeline...")
    
    # Set the data directory (use standardized datasets from preprocessing step)
    data_dir = "standardized_datasets"
    print(f"\nChecking data directory: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found!")
        print("Run preprocessing.py first to create standardized datasets.")
        exit(1)
    
    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not dataset_dirs:
        print(f"No dataset directories found in {data_dir}!")
        exit(1)
    
    print(f"\nFound {len(dataset_dirs)} standardized datasets:")
    for d in dataset_dirs:
        print(f"- {d}")
    
    # Deduplicate each dataset
    deduplication_results = []
    deduplicated_paths = {}
    
    for dataset_name in dataset_dirs:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_path = os.path.join(data_dir, dataset_name)
        
        # Deduplicate dataset
        output_path, stats, removed_images, kept_images = deduplicate_dataset(
            dataset_path, dataset_name)
        
        # Add deduplicated path to dictionary
        deduplicated_paths[dataset_name] = output_path
        
        # Add results to list
        deduplication_results.append({
            'dataset_name': dataset_name,
            'deduplication_stats': stats
        })
        
        # Visualize results for this dataset
        print("\nGenerating visualization for deduplication...")
        plot_duplicate_summary(
            dataset_name,
            stats['total_original_images'],
            stats['images_to_remove'],
            stats['total_duplicate_groups']
        )
        
        # Show class distribution before and after deduplication
        plot_class_distribution(
            dataset_name,
            stats['original_class_counts'],
            stats['kept_class_counts'],
            title=f"{dataset_name} Class Distribution: Original vs Deduplicated"
        )
    
    # Convert deduplication results to DataFrame
    deduplication_df = pd.DataFrame(deduplication_results)
    
    # Save deduplication stats to Excel
    deduplication_df.to_excel(
        os.path.join("excel", "deduplication_results.xlsx"), 
        index=False
    )
    print("\nSaved deduplication results to excel/deduplication_results.xlsx")
    
    # Run EDA on deduplicated datasets
    print("\nPerforming EDA on deduplicated datasets...")
    deduplicated_eda_results = []
    
    for dataset_name, dataset_path in deduplicated_paths.items():
        print(f"\nPerforming EDA on deduplicated {dataset_name}...")
        deduplicated_eda = perform_image_eda(dataset_path, f"Deduplicated_{dataset_name}", "plots")
        deduplicated_eda_results.append(deduplicated_eda)
        print(f"Completed EDA for deduplicated {dataset_name}")
    
    # Convert deduplicated EDA results to DataFrame
    deduplicated_eda_df = pd.DataFrame(deduplicated_eda_results)
    
    # Save deduplicated EDA results
    deduplicated_eda_df.to_excel(
        os.path.join("excel", "deduplicated_datasets_eda.xlsx"), 
        index=False
    )
    print("\nSaved deduplicated datasets EDA to excel/deduplicated_datasets_eda.xlsx")
    
    # Return the deduplicated paths and deduplication results
    return deduplicated_paths, deduplication_df, deduplicated_eda_results, deduplicated_eda_df

if __name__ == "__main__":
    # Run the deduplication pipeline
    deduplicated_paths, deduplication_df, deduplicated_eda_results, deduplicated_eda_df = main()
    
    print("\nDeduplication complete! To continue with data augmentation, run:")
    print("  python data_augmentation.py") 