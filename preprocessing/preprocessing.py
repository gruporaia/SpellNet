#!/usr/bin/env python3
"""
Sign Language Datasets Preprocessing Module

This module handles the initial preprocessing of sign language image datasets:
- Hardware detection and optimization for parallel processing
- Metadata analysis of original datasets (dimensions, formats, class distribution)
- Standardization of images to 224x224 PNG format with optimized resizing

The module creates standardized datasets ready for deduplication and further processing.
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
import seaborn as sns
import random
import cv2
from collections import defaultdict
import warnings
import multiprocessing
from functools import partial
import concurrent.futures
import psutil
import threading
from queue import Queue
from threading import Thread
import time
warnings.filterwarnings('ignore')

# Import EDA functionality from the eda module
from eda import perform_image_eda

# Detect hardware and set optimization parameters
CPU_COUNT = multiprocessing.cpu_count()
TOTAL_RAM = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
OPTIMAL_WORKERS = min(CPU_COUNT - 1, 10)  # Leave one core free for system
BATCH_SIZE = int(max(100, min(5000, TOTAL_RAM * 200)))  # Scale batch size with RAM

# Check if OpenCV can use GPU
try:
    cv2.cuda.getCudaEnabledDeviceCount()
    HAS_GPU = cv2.cuda.getCudaEnabledDeviceCount() > 0
except:
    HAS_GPU = False

# Create necessary directories
os.makedirs('standardized_datasets', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('excel', exist_ok=True)

# Thread-based file reader (faster than sequential I/O)
class AsyncImageLoader:
    def __init__(self, file_list, max_queue_size=100):
        self.file_list = file_list
        self.queue = Queue(maxsize=max_queue_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.fill_queue, daemon=True).start()
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

# Optimized function to resize images in parallel
def resize_images_parallel(image_paths, output_dir, target_size=(224, 224), 
                           format='PNG', num_workers=None, batch_size=1000):
    """
    Resize images in parallel using multiple workers
    
    Args:
        image_paths (list): List of image paths
        output_dir (str): Output directory
        target_size (tuple): Target size (width, height)
        format (str): Output format
        num_workers (int): Number of worker processes
        batch_size (int): Batch size for processing
    """
    if num_workers is None:
        num_workers = OPTIMAL_WORKERS
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare batches
    batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    
    # Process each batch with multiple workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch_idx, batch in enumerate(batches):
            future = executor.submit(
                _process_resize_batch, 
                batch, 
                batch_idx, 
                len(batches), 
                output_dir, 
                target_size, 
                format
            )
            futures.append(future)
        
        # Show progress
        with tqdm(total=len(image_paths), desc="Resizing images") as pbar:
            for future in concurrent.futures.as_completed(futures):
                processed = future.result()
                pbar.update(processed)
    
    return output_dir

def _process_resize_batch(batch, batch_idx, total_batches, output_dir, target_size, format):
    """Process a batch of images for resizing"""
    count = 0
    # Start async loader
    loader = AsyncImageLoader(batch).start()
    
    while True:
        file_path, img = loader.get()
        if file_path is None:
            break
            
        try:
            # Get output path
            rel_path = os.path.relpath(file_path, start=os.path.dirname(output_dir))
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Resize with high-quality downsampling
            if img.width > target_size[0] or img.height > target_size[1]:
                img = img.resize(target_size, Image.LANCZOS)
            else:
                img = img.resize(target_size, Image.BICUBIC)
                
            # Save image
            img.save(output_path, format=format)
            count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    loader.stop()
    return count

# Function to analyze metadata of image datasets
def analyze_image_dataset_metadata(dataset_path, dataset_name):
    """
    Analyze metadata of an image dataset
    
    Args:
        dataset_path (str): Path to the dataset
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Dictionary with metadata analysis
    """
    print(f"Analyzing metadata for {dataset_name}...")
    
    # Find all image files in the dataset
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    
    if len(all_images) == 0:
        print(f"  No images found in {dataset_path}")
        return {
            'dataset_name': dataset_name,
            'total_images': 0,
            'image_formats': [],
            'classes': [],
            'images_per_class': {},
            'avg_width': 0,
            'avg_height': 0,
            'min_width': 0,
            'min_height': 0,
            'max_width': 0,
            'max_height': 0,
            'avg_aspect_ratio': 0,
            'color_modes': [],
        }
    
    print(f"  Found {len(all_images)} images")
    
    # Sample a subset of images for detailed analysis
    sample_size = min(1000, len(all_images))
    sampled_images = np.random.choice(all_images, sample_size, replace=False)
    
    # Initialize variables for metadata
    formats = set()
    widths = []
    heights = []
    color_modes = set()
    channels = []
    
    # Use thread-based loading for faster I/O
    loader = AsyncImageLoader(sampled_images).start()
    
    # Analyze sampled images
    print(f"  Analyzing a sample of {sample_size} images")
    with tqdm(total=sample_size, desc="Analyzing images") as pbar:
        while True:
            img_path, img = loader.get()
            if img_path is None:
                break
                
            try:
                formats.add(img.format)
                widths.append(img.width)
                heights.append(img.height)
                color_modes.add(img.mode)
                
                # Determine number of channels
                if img.mode == 'RGB':
                    channels.append(3)
                elif img.mode == 'RGBA':
                    channels.append(4)
                elif img.mode == 'L':
                    channels.append(1)
                else:
                    channels.append(0)  # Unknown
                
                pbar.update(1)
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                pbar.update(1)
    
    loader.stop()
    
    # Count images per class (assuming class is the parent directory name)
    classes = {}
    for img_path in all_images:
        class_name = os.path.basename(os.path.dirname(img_path))
        classes[class_name] = classes.get(class_name, 0) + 1
    
    # Compile metadata
    metadata = {
        'dataset_name': dataset_name,
        'total_images': len(all_images),
        'image_formats': list(formats),
        'classes': list(classes.keys()),
        'num_classes': len(classes),
        'images_per_class': classes,
        'avg_width': np.mean(widths) if widths else 0,
        'avg_height': np.mean(heights) if heights else 0,
        'min_width': np.min(widths) if widths else 0,
        'min_height': np.min(heights) if heights else 0,
        'max_width': np.max(widths) if widths else 0,
        'max_height': np.max(heights) if heights else 0,
        'avg_aspect_ratio': np.mean([w/h for w, h in zip(widths, heights)]) if widths and heights else 0,
        'color_modes': list(color_modes),
        'avg_channels': np.mean(channels) if channels else 0,
    }
    
    print(f"  Metadata analysis complete for {dataset_name}")
    print(f"  Number of classes: {metadata['num_classes']}")
    print(f"  Average dimensions: {metadata['avg_width']}x{metadata['avg_height']}")
    print(f"  Color modes: {', '.join(metadata['color_modes'])}")
    
    return metadata

# Function to standardize datasets
def standardize_datasets(metadata_df, datasets_paths):
    """
    Standardize datasets based on metadata analysis
    
    Args:
        metadata_df (DataFrame): DataFrame with metadata analysis
        datasets_paths (dict): Dictionary with dataset paths
        
    Returns:
        dict: Dictionary with standardized dataset paths
    """
    print("\nStandardizing datasets...")
    
    # Check if image formats need to be standardized
    formats = metadata_df['image_formats'].apply(lambda x: set(x))
    all_formats = set()
    for f in formats:
        all_formats.update(f)
    
    print(f"Found image formats: {all_formats}")
    
    # Check if all datasets have the same classes
    classes = metadata_df['classes'].apply(lambda x: set(x))
    
    standardized_paths = {}
    
    # Standardize each dataset
    for dataset_name, dataset_path in datasets_paths.items():
        print(f"\nStandardizing dataset: {dataset_name}")
        std_dataset_path = os.path.join("standardized_datasets", dataset_name)
        os.makedirs(std_dataset_path, exist_ok=True)
        standardized_paths[dataset_name] = std_dataset_path
        
        # Find the classes in this dataset
        this_dataset_classes = set(metadata_df[metadata_df['dataset_name'] == dataset_name]['classes'].iloc[0])
        
        # Create class directories
        for class_name in this_dataset_classes:
            os.makedirs(os.path.join(std_dataset_path, class_name), exist_ok=True)
        
        # Find all image files in the dataset
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
            all_images.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
        
        print(f"  Found {len(all_images)} images to standardize")
        
        # Organize images by class
        images_by_class = defaultdict(list)
        for img_path in all_images:
            parts = Path(img_path).parts
            class_name = None
            for part in parts:
                if part in this_dataset_classes:
                    class_name = part
                    break
            
            if class_name is not None:
                images_by_class[class_name].append(img_path)
        
        # Process each class in parallel
        start_time = time.time()
        num_workers = OPTIMAL_WORKERS
        batch_size = BATCH_SIZE
        total_processed = 0
        
        for class_name, class_images in images_by_class.items():
            class_output_dir = os.path.join(std_dataset_path, class_name)
            
            # Process this class's images
            print(f"  Processing class {class_name}: {len(class_images)} images")
            
            # Prepare output paths and create dictionaries for mapping
            output_paths = []
            for img_path in class_images:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(class_output_dir, f"{base_name}.png")
                output_paths.append(output_path)
            
            # Create batches for parallel processing
            image_batches = [class_images[i:i+batch_size] for i in range(0, len(class_images), batch_size)]
            
            processed_count = 0
            with tqdm(total=len(class_images), desc=f"Standardizing {class_name}") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit batches for processing
                    futures = []
                    for batch in image_batches:
                        future = executor.submit(process_image_batch, batch, class_output_dir, (224, 224))
                        futures.append(future)
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        batch_count = future.result()
                        processed_count += batch_count
                        pbar.update(batch_count)
            
            total_processed += processed_count
        
        elapsed_time = time.time() - start_time
        images_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
        print(f"  Dataset {dataset_name} standardized to 224x224x3 PNG format")
        print(f"  Processed {total_processed} images in {elapsed_time:.1f} seconds ({images_per_second:.1f} images/sec)")
    
    print("\nAll datasets standardized")
    return standardized_paths

def process_image_batch(image_paths, output_dir, target_size=(224, 224)):
    """
    Process a batch of images
    
    Args:
        image_paths (list): List of image paths
        output_dir (str): Output directory
        target_size (tuple): Target size (width, height)
        
    Returns:
        int: Number of processed images
    """
    count = 0
    
    # Use GPU acceleration if available
    use_gpu = HAS_GPU and len(image_paths) > 10  # Only use GPU for larger batches
    
    if use_gpu:
        # Process with GPU
        try:
            return process_image_batch_gpu(image_paths, output_dir, target_size)
        except Exception as e:
            print(f"  GPU processing failed, falling back to CPU: {e}")
            use_gpu = False
    
    # CPU processing
    if not use_gpu:
        for img_path in image_paths:
            try:
                # Open image
                img = Image.open(img_path).convert('RGB')
                
                # Resize to target size with high quality
                if img.width > target_size[0] or img.height > target_size[1]:
                    img_resized = img.resize(target_size, Image.LANCZOS)
                else:
                    img_resized = img.resize(target_size, Image.BICUBIC)
                
                # Create output filename
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.png")
                
                # Save as PNG
                img_resized.save(output_path, format="PNG")
                count += 1
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
    
    return count

def process_image_batch_gpu(image_paths, output_dir, target_size=(224, 224)):
    """
    Process a batch of images using GPU acceleration
    
    Args:
        image_paths (list): List of image paths
        output_dir (str): Output directory
        target_size (tuple): Target size (width, height)
        
    Returns:
        int: Number of processed images
    """
    if not HAS_GPU:
        return process_image_batch(image_paths, output_dir, target_size)
    
    count = 0
    for img_path in image_paths:
        try:
            # Read image with OpenCV
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                continue
                
            # Convert to RGB (OpenCV uses BGR)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img_cv)
            
            # Resize on GPU
            gpu_resized = cv2.cuda.resize(gpu_img, target_size)
            
            # Download result
            resized_cv = gpu_resized.download()
            
            # Convert back to BGR for saving
            resized_cv = cv2.cvtColor(resized_cv, cv2.COLOR_RGB2BGR)
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            # Save as PNG
            cv2.imwrite(output_path, resized_cv)
            count += 1
        except Exception as e:
            # Fall back to CPU for this image
            try:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize(target_size, Image.LANCZOS)
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.png")
                img_resized.save(output_path, format="PNG")
                count += 1
            except Exception as inner_e:
                print(f"  Error processing {img_path}: {e}, {inner_e}")
    
    return count

def main():
    """Main function to run the preprocessing pipeline"""
    print("Starting ASL and Libras Datasets Preprocessing Pipeline...")
    
    # Set the data directory
    data_dir = "original_dataset(2)"
    print(f"\nChecking data directory: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found!")
        exit(1)
    
    # Get all dataset directories
    dataset_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not dataset_dirs:
        print(f"No dataset directories found in {data_dir}!")
        exit(1)
    
    print(f"\nFound {len(dataset_dirs)} datasets:")
    for d in dataset_dirs:
        print(f"- {d}")
    
    print("\nStarting metadata analysis...")
    # Analyze metadata for each dataset
    metadata_results = []
    dataset_paths = {}
    
    for dataset_name in dataset_dirs:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_path = os.path.join(data_dir, dataset_name)
        dataset_paths[dataset_name] = dataset_path
        metadata = analyze_image_dataset_metadata(dataset_path, dataset_name)
        metadata_results.append(metadata)
        print(f"Completed metadata analysis for {dataset_name}")
    
    print("\nConverting metadata results to DataFrame...")
    # Convert metadata results to DataFrame
    metadata_df = pd.DataFrame(metadata_results)
    
    # Save metadata to Excel
    metadata_df.to_excel(os.path.join("excel", "original_datasets_metadata.xlsx"), index=False)
    print("Saved original datasets metadata to excel/original_datasets_metadata.xlsx")
    
    print("\nStarting dataset standardization...")
    # Standardize datasets to 224x224x3 PNG format
    standardized_paths = standardize_datasets(metadata_df, dataset_paths)
    
    print("\nStarting EDA on standardized datasets...")
    # Perform EDA on standardized datasets using the imported function
    original_eda_results = []
    for dataset_name, dataset_path in standardized_paths.items():
        print(f"\nPerforming EDA on standardized {dataset_name}...")
        eda = perform_image_eda(dataset_path, dataset_name, "plots")
        original_eda_results.append(eda)
        print(f"Completed EDA for standardized {dataset_name}")
    
    print("\nConverting EDA results to DataFrame...")
    # Convert EDA results to DataFrame
    original_eda_df = pd.DataFrame(original_eda_results)
    
    # Save original EDA results
    original_eda_df.to_excel(os.path.join("excel", "standardized_datasets_eda.xlsx"), index=False)
    print("Saved standardized datasets EDA to excel/standardized_datasets_eda.xlsx")
    
    # Return the standardized paths and metadata for deduplication
    return standardized_paths, metadata_df, original_eda_results, original_eda_df

if __name__ == "__main__":
    print(f"Hardware detected: {CPU_COUNT} CPU cores, {TOTAL_RAM:.1f}GB RAM")
    print(f"OpenCV GPU acceleration: {'Available' if HAS_GPU else 'Not available'}")
    print(f"Using {OPTIMAL_WORKERS} worker threads and batch size of {BATCH_SIZE}")
    
    # Run the preprocessing pipeline
    standardized_paths, metadata_df, original_eda_results, original_eda_df = main()
    
    print("\nPreprocessing complete! To continue with deduplication, run:")
    print("  python deduplication.py") 