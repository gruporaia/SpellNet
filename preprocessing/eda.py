#!/usr/bin/env python3
"""
Sign Language Datasets EDA (Exploratory Data Analysis) Module

This module handles the exploratory data analysis of sign language image datasets:
- Analysis of image characteristics (brightness, contrast, edge density, etc.)
- Generation of basic statistical metrics for image datasets
- Creation of distribution plots for image characteristics
- Analysis of class distributions within datasets

The module provides insight into dataset characteristics for better preprocessing decisions.
"""
#Author: MatheusHRV

import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import warnings
from tqdm import tqdm
import seaborn as sns

warnings.filterwarnings('ignore')

# Function to perform exploratory data analysis on images
def perform_image_eda(dataset_path, dataset_name, output_dir="plots"):
    """
    Perform exploratory data analysis on an image dataset
    
    Args:
        dataset_path (str): Path to the dataset
        dataset_name (str): Name of the dataset
        output_dir (str): Directory to save plots
        
    Returns:
        dict: Dictionary with EDA results
    """
    print(f"\nPerforming EDA for {dataset_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files in the dataset
    all_images = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
    
    if len(all_images) == 0:
        print(f"  No images found in {dataset_path}")
        return {
            'dataset_name': dataset_name,
            'total_images': 0
        }
    
    print(f"  Found {len(all_images)} images for analysis")
    
    # Sample images for analysis
    sample_size = min(1000, len(all_images))
    sampled_images = np.random.choice(all_images, sample_size, replace=False)
    
    # Initialize data collection
    brightness_values = []
    contrast_values = []
    saturation_values = []
    colors_per_image = []
    class_counts = {}
    edge_density = []
    
    # For color distribution
    r_means = []
    g_means = []
    b_means = []
    r_stds = []
    g_stds = []
    b_stds = []
    
    # Analyze each image
    print(f"  Analyzing a sample of {sample_size} images")
    for img_path in tqdm(sampled_images, desc="Analyzing images for EDA"):
        try:
            # Get class from path
            class_name = os.path.basename(os.path.dirname(img_path))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Open and analyze image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Calculate brightness (average pixel intensity)
            brightness = np.mean(img_array)
            brightness_values.append(brightness)
            
            # Calculate contrast (standard deviation of pixel intensities)
            contrast = np.std(img_array)
            contrast_values.append(contrast)
            
            # Calculate color distributions
            r_channel = img_array[:,:,0]
            g_channel = img_array[:,:,1]
            b_channel = img_array[:,:,2]
            
            r_means.append(np.mean(r_channel))
            g_means.append(np.mean(g_channel))
            b_means.append(np.mean(b_channel))
            
            r_stds.append(np.std(r_channel))
            g_stds.append(np.std(g_channel))
            b_stds.append(np.std(b_channel))
            
            # Calculate saturation
            r, g, b = r_channel/255.0, g_channel/255.0, b_channel/255.0
            max_rgb = np.maximum(np.maximum(r, g), b)
            min_rgb = np.minimum(np.minimum(r, g), b)
            saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-10))
            saturation_values.append(saturation)
            
            # Count unique colors
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
            colors_per_image.append(unique_colors)
            
            # Edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density.append(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
            
        except Exception as e:
            print(f"  Error processing {img_path} for EDA: {e}")
    
    # Create EDA results dictionary
    eda_results = {
        'dataset_name': dataset_name,
        'total_images': len(all_images),
        'num_samples_analyzed': len(brightness_values),
        'num_classes': len(class_counts),
        'avg_brightness': np.mean(brightness_values) if brightness_values else 0,
        'std_brightness': np.std(brightness_values) if brightness_values else 0,
        'avg_contrast': np.mean(contrast_values) if contrast_values else 0,
        'std_contrast': np.std(contrast_values) if contrast_values else 0,
        'avg_saturation': np.mean(saturation_values) if saturation_values else 0,
        'std_saturation': np.std(saturation_values) if saturation_values else 0,
        'avg_unique_colors': np.mean(colors_per_image) if colors_per_image else 0,
        'std_unique_colors': np.std(colors_per_image) if colors_per_image else 0,
        'avg_edge_density': np.mean(edge_density) if edge_density else 0,
        'std_edge_density': np.std(edge_density) if edge_density else 0,
        'avg_r_mean': np.mean(r_means) if r_means else 0,
        'avg_g_mean': np.mean(g_means) if g_means else 0,
        'avg_b_mean': np.mean(b_means) if b_means else 0,
        'avg_r_std': np.mean(r_stds) if r_stds else 0,
        'avg_g_std': np.mean(g_stds) if g_stds else 0,
        'avg_b_std': np.mean(b_stds) if b_stds else 0,
        'class_distribution': class_counts,
    }
    
    # Generate plots
    if brightness_values:
        # Brightness distribution
        plt.figure(figsize=(10, 6))
        plt.hist(brightness_values, bins=50)
        plt.title(f"{dataset_name} - Brightness Distribution")
        plt.xlabel("Brightness")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_brightness.png"))
        plt.close()
        
        # Contrast distribution
        plt.figure(figsize=(10, 6))
        plt.hist(contrast_values, bins=50)
        plt.title(f"{dataset_name} - Contrast Distribution")
        plt.xlabel("Contrast")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_contrast.png"))
        plt.close()
        
        # Class distribution
        plt.figure(figsize=(12, 8))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title(f"{dataset_name} - Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_class_dist.png"))
        plt.close()
        
        # Color channel distributions
        plt.figure(figsize=(12, 8))
        sns.kdeplot(r_means, color='r', label='Red Channel')
        sns.kdeplot(g_means, color='g', label='Green Channel')
        sns.kdeplot(b_means, color='b', label='Blue Channel')
        plt.title(f"{dataset_name} - Color Channel Distributions")
        plt.xlabel("Mean Channel Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_color_dist.png"))
        plt.close()
        
        # Edge density distribution
        plt.figure(figsize=(10, 6))
        plt.hist(edge_density, bins=50)
        plt.title(f"{dataset_name} - Edge Density Distribution")
        plt.xlabel("Edge Density")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_edge_density.png"))
        plt.close()
    
    print(f"  EDA complete for {dataset_name}")
    return eda_results

def generate_eda_report(eda_results, output_excel="excel/eda_report.xlsx"):
    """
    Generate a report from EDA results
    
    Args:
        eda_results (list): List of EDA result dictionaries
        output_excel (str): Path to save Excel report
        
    Returns:
        DataFrame: DataFrame with EDA results
    """
    print("\nGenerating EDA report...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    
    # Convert to DataFrame
    eda_df = pd.DataFrame(eda_results)
    
    # Save to Excel
    eda_df.to_excel(output_excel, index=False)
    print(f"  EDA report saved to {output_excel}")
    
    return eda_df

def run_eda_pipeline(dataset_paths, output_excel="excel/eda_report.xlsx", output_dir="plots"):
    """
    Run the EDA pipeline on multiple datasets
    
    Args:
        dataset_paths (dict): Dictionary mapping dataset names to paths
        output_excel (str): Path to save Excel report
        output_dir (str): Directory to save plots
        
    Returns:
        tuple: (DataFrame with EDA results, list of EDA result dictionaries)
    """
    print("Starting EDA Pipeline...")
    
    # Create directories
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run EDA on each dataset
    eda_results = []
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"\nRunning EDA on {dataset_name}...")
        eda = perform_image_eda(dataset_path, dataset_name, output_dir)
        eda_results.append(eda)
        print(f"  Completed EDA for {dataset_name}")
    
    # Generate report
    eda_df = generate_eda_report(eda_results, output_excel)
    
    print("\nEDA pipeline complete!")
    return eda_df, eda_results

if __name__ == "__main__":
    # This script can be run standalone on specified datasets
    # Example usage:
    print("This module provides EDA functionality for sign language datasets.")
    print("Import and use the functions in your pipeline or specify datasets to analyze.")
    
    # Check if standardized datasets exist
    std_dir = "standardized_datasets"
    if os.path.exists(std_dir):
        dataset_paths = {}
        for d in os.listdir(std_dir):
            if os.path.isdir(os.path.join(std_dir, d)) and not d.endswith(("_no_duplicates", "_augmented")):
                dataset_paths[d] = os.path.join(std_dir, d)
        
        if dataset_paths:
            print(f"\nFound {len(dataset_paths)} standardized datasets:")
            for name, path in dataset_paths.items():
                print(f"  - {name}: {path}")
            
            # Ask if user wants to analyze these datasets
            response = input("\nDo you want to run EDA on these datasets? (y/n): ")
            if response.lower() == 'y':
                run_eda_pipeline(dataset_paths) 