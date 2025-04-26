#!/usr/bin/env python3
"""
Sign Language Datasets Augmentation Module

This module handles the augmentation of sign language image datasets:
- Image augmentation with multiple transformation techniques
- Balanced augmentation to equalize class distributions
- Generation of reports on augmented dataset characteristics
- Management of augmentation pipelines for reproducibility

The module creates augmented datasets ready for machine learning training.
"""
#Author: MatheusHRV

import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import random
import warnings
import multiprocessing
import concurrent.futures
import albumentations as A
import time
warnings.filterwarnings('ignore')

# Import from preprocessing and deduplication for pipeline connection
from preprocessing import (
    perform_image_eda,
    process_image_batch,
    HAS_GPU, 
    CPU_COUNT, 
    TOTAL_RAM, 
    OPTIMAL_WORKERS, 
    BATCH_SIZE
)

# Define augmentation transformations
def get_augmentation_pipeline(severity='medium'):
    """
    Get an augmentation pipeline based on severity level
    
    Args:
        severity (str): Severity level of augmentation ('light', 'medium', 'heavy')
        
    Returns:
        A.Compose: Albumentation composition of transformations
    """
    if severity == 'light':
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        ])
    elif severity == 'medium':
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.7),
            A.ElasticTransform(alpha=0.5, sigma=50, alpha_affine=50, p=0.3),
            A.RandomShadow(p=0.3),
        ])
    else:  # heavy
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.GaussianBlur(blur_limit=(3, 9), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=45, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.8),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.RandomShadow(p=0.5),
            A.RandomFog(p=0.3),
            A.ISONoise(p=0.3),
        ])

def augment_image(img_path, transform, output_path, aug_index):
    """
    Apply augmentation to a single image
    
    Args:
        img_path (str): Path to input image
        transform (A.Compose): Albumentation transformation to apply
        output_path (str): Path to save augmented image
        aug_index (int): Index of augmentation for filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            return False
            
        # Convert to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        augmented = transform(image=img)
        img_augmented = augmented['image']
        
        # Convert back to BGR for saving
        img_augmented = cv2.cvtColor(img_augmented, cv2.COLOR_RGB2BGR)
        
        # Save augmented image
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_file = os.path.join(output_path, f"{base_name}_aug{aug_index}.png")
        cv2.imwrite(output_file, img_augmented)
        
        return True
    except Exception as e:
        print(f"Error augmenting {img_path}: {e}")
        return False

def process_augmentation_batch(batch_data):
    """
    Process a batch of images for augmentation
    
    Args:
        batch_data (tuple): (image_paths, output_dir, transforms_per_image, transform)
        
    Returns:
        int: Number of augmented images generated
    """
    image_paths, output_dir, transforms_per_image, transform = batch_data
    count = 0
    
    for img_path in image_paths:
        # Create output directory
        class_name = os.path.basename(os.path.dirname(img_path))
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Apply multiple transformations per image
        for i in range(transforms_per_image):
            if augment_image(img_path, transform, class_output_dir, i+1):
                count += 1
                
    return count

def augment_dataset(dataset_path, output_path, severity='medium', transforms_per_image=3, balance_classes=True, num_workers=None, batch_size=None):
    """
    Augment a dataset of images
    
    Args:
        dataset_path (str): Path to input dataset
        output_path (str): Path to save augmented dataset
        severity (str): Severity level of augmentation
        transforms_per_image (int): Number of augmented versions per image
        balance_classes (bool): Whether to balance classes
        num_workers (int): Number of worker processes
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (output path, stats dictionary)
    """
    print(f"\nAugmenting dataset from {dataset_path}...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Find all PNG images in the dataset
    all_images = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
    
    if len(all_images) == 0:
        print(f"  No images found in {dataset_path}")
        return output_path, {'total_images': 0, 'augmented_images': 0}
    
    # First, copy original images to output directory
    print(f"  Copying {len(all_images)} original images...")
    for img_path in tqdm(all_images, desc="Copying original images"):
        try:
            # Get class name
            class_name = os.path.basename(os.path.dirname(img_path))
            
            # Create class directory if it doesn't exist
            os.makedirs(os.path.join(output_path, class_name), exist_ok=True)
            
            # Copy image
            shutil.copy2(img_path, os.path.join(output_path, class_name, os.path.basename(img_path)))
        except Exception as e:
            print(f"  Error copying {img_path}: {e}")
    
    # Get class distribution
    class_counts = defaultdict(int)
    for img_path in all_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        class_counts[class_name] += 1
    
    # Calculate augmentation per class if balancing
    if balance_classes:
        max_class_count = max(class_counts.values())
        
        # For each class, calculate how many augmentations per image to reach the max
        augmentations_per_class = {}
        for class_name, count in class_counts.items():
            if count < max_class_count:
                # Calculate augmentations needed per original image
                total_needed = max_class_count - count
                augs_per_img = max(1, min(10, total_needed // count + 1))
                augmentations_per_class[class_name] = augs_per_img
            else:
                # For the largest class, just add some variation
                augmentations_per_class[class_name] = 1
                
        print(f"  Balancing classes to {max_class_count} images each")
    else:
        # Use the same number of transforms for all classes
        augmentations_per_class = {class_name: transforms_per_image for class_name in class_counts.keys()}
    
    # Create augmentation pipeline
    transform = get_augmentation_pipeline(severity)
    
    # Group images by class
    images_by_class = defaultdict(list)
    for img_path in all_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        images_by_class[class_name].append(img_path)
    
    # Set up parallel processing
    if num_workers is None:
        num_workers = OPTIMAL_WORKERS
    
    if batch_size is None:
        batch_size = min(100, max(10, len(all_images) // (num_workers * 2)))
    
    # Create batches for parallel processing
    batches = []
    for class_name, class_images in images_by_class.items():
        transforms_count = augmentations_per_class[class_name]
        # Split class images into batches
        for i in range(0, len(class_images), batch_size):
            batch_images = class_images[i:i+batch_size]
            batches.append((batch_images, output_path, transforms_count, transform))
    
    # Process batches in parallel
    print(f"  Augmenting images with {severity} severity using {num_workers} workers...")
    total_augmented = 0
    
    with tqdm(total=len(all_images), desc="Generating augmentations") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_augmentation_batch, batch) for batch in batches]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_count = future.result()
                    total_augmented += batch_count
                    pbar.update(batch_count // 3)  # Approximate update based on avg transforms
                except Exception as e:
                    print(f"  Error in augmentation batch: {e}")
    
    # Gather statistics
    final_images = glob.glob(os.path.join(output_path, '**', '*.png'), recursive=True)
    final_class_counts = defaultdict(int)
    
    for img_path in final_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        final_class_counts[class_name] += 1
    
    stats = {
        'original_images': len(all_images),
        'augmented_images': total_augmented,
        'total_images': len(final_images),
        'original_class_counts': dict(class_counts),
        'final_class_counts': dict(final_class_counts)
    }
    
    print(f"  Dataset augmentation complete:")
    print(f"    Original images: {len(all_images)}")
    print(f"    Generated augmentations: {total_augmented}")
    print(f"    Total final images: {len(final_images)}")
    
    # Plot the class distribution before and after augmentation
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'Class': list(final_class_counts.keys()),
        'Original': [class_counts[cls] for cls in final_class_counts.keys()],
        'Augmented': [final_class_counts[cls] for cls in final_class_counts.keys()]
    })
    
    # Sort by original count
    df_plot = df_plot.sort_values('Original', ascending=False)
    
    # Create a grouped bar chart
    plt.bar(df_plot['Class'], df_plot['Original'], label='Original')
    plt.bar(df_plot['Class'], df_plot['Augmented'] - df_plot['Original'], 
            bottom=df_plot['Original'], label='Augmented')
    
    plt.title('Class Distribution: Original vs Augmented')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("plots", "augmentation_class_distribution.png"))
    plt.close()
    
    return output_path, stats

def main():
    """Main function to run the augmentation pipeline"""
    print("Starting ASL and Libras Datasets Augmentation Pipeline...")
    
    # Try to import deduplication results
    try:
        from deduplication import main as dedup_main
        
        # Run deduplication if needed
        print("\nRunning deduplication step...")
        dedup_paths, dedup_eda_df = dedup_main()
    except Exception as e:
        print(f"Error importing deduplication results: {e}")
        print("Will search for deduplicated datasets directly...")
        
        # Try to find deduplicated datasets
        dedup_paths = {}
        datasets_dir = "standardized_datasets"
        for d in os.listdir(datasets_dir):
            if "_no_duplicates" in d and os.path.isdir(os.path.join(datasets_dir, d)):
                dataset_name = d.replace("_no_duplicates", "")
                dedup_paths[dataset_name] = os.path.join(datasets_dir, d)
        
        if not dedup_paths:
            print("No deduplicated datasets found. Please run deduplication.py first.")
            exit(1)
    
    # Perform augmentation on each deduplicated dataset
    augmented_paths = {}
    augmentation_stats = {}
    
    for dataset_name, dataset_path in dedup_paths.items():
        print(f"\nProcessing dataset: {dataset_name}")
        output_path = os.path.join("standardized_datasets", f"{dataset_name}_augmented")
        
        # Process dataset
        try:
            augmented_path, stats = augment_dataset(
                dataset_path,
                output_path,
                severity='medium',
                transforms_per_image=3,
                balance_classes=True
            )
            augmented_paths[dataset_name] = augmented_path
            augmentation_stats[dataset_name] = stats
        except Exception as e:
            print(f"Error augmenting {dataset_name}: {e}")
            print(f"Skipping augmentation for {dataset_name}")
    
    # Generate summary report
    print("\nAugmentation Summary:")
    for dataset_name, stats in augmentation_stats.items():
        print(f"  {dataset_name}:")
        print(f"    Original images: {stats['original_images']}")
        print(f"    Generated augmentations: {stats['augmented_images']}")
        print(f"    Total images: {stats['total_images']}")
        print(f"    Augmentation factor: {stats['total_images'] / stats['original_images']:.2f}x")
    
    # Save augmentation stats to Excel
    try:
        # Convert stats to DataFrame
        stats_list = []
        for dataset_name, stats in augmentation_stats.items():
            stats_data = {
                'dataset_name': dataset_name,
                'original_images': stats['original_images'],
                'augmented_images': stats['augmented_images'],
                'total_images': stats['total_images'],
                'augmentation_factor': stats['total_images'] / stats['original_images'],
            }
            
            # Add class counts
            for class_name, count in stats['original_class_counts'].items():
                stats_data[f'original_{class_name}'] = count
                
            for class_name, count in stats['final_class_counts'].items():
                stats_data[f'final_{class_name}'] = count
                
            stats_list.append(stats_data)
            
        # Create DataFrame and save to Excel
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_excel(os.path.join("excel", "augmentation_stats.xlsx"), index=False)
        print("\nSaved augmentation statistics to excel/augmentation_stats.xlsx")
    except Exception as e:
        print(f"Error saving augmentation stats: {e}")
    
    return augmented_paths, augmentation_stats

if __name__ == "__main__":
    # Run the augmentation pipeline
    augmented_paths, augmentation_stats = main()
    
    print("\nAugmentation complete! Datasets are ready for model training.")
    print("Final datasets can be found in:")
    for dataset_name, path in augmented_paths.items():
        print(f"  - {path}") 