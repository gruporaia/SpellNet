#!/usr/bin/env python3
"""
Sign Language Datasets Visualization Module

This module handles the visualization of sign language dataset analyses:
- Comparative visualization between original, deduplicated, and augmented datasets
- Generation of summary plots for dataset characteristics
- Visualization of class distributions across preprocessing stages
- Creation of reports and visualizations for dataset quality metrics

The module provides visual insights into dataset transformations for better understanding.
"""
#Author: MatheusHRV

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

def plot_class_distribution(dataset_paths, output_dir="plots", filename="class_distribution.png"):
    """
    Plot class distribution across multiple datasets
    
    Args:
        dataset_paths (dict): Dictionary mapping dataset names to paths
        output_dir (str): Directory to save plots
        filename (str): Filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    print("Generating class distribution plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect class counts for each dataset
    dataset_class_counts = {}
    all_classes = set()
    
    for dataset_name, dataset_path in dataset_paths.items():
        class_counts = defaultdict(int)
        try:
            for dir_path in os.listdir(dataset_path):
                full_dir_path = os.path.join(dataset_path, dir_path)
                if os.path.isdir(full_dir_path):
                    num_images = len([f for f in os.listdir(full_dir_path) if f.endswith('.png')])
                    class_counts[dir_path] = num_images
                    all_classes.add(dir_path)
            dataset_class_counts[dataset_name] = class_counts
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
    
    # Create DataFrame for plotting
    all_classes = sorted(list(all_classes))
    data = []
    
    for dataset_name, class_counts in dataset_class_counts.items():
        for class_name in all_classes:
            data.append({
                'Dataset': dataset_name,
                'Class': class_name,
                'Count': class_counts.get(class_name, 0)
            })
    
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x='Class', y='Count', hue='Dataset', data=df)
    plt.title('Class Distribution Across Datasets')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.legend(title='Dataset')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"  Class distribution plot saved to {output_path}")
    return output_path

def plot_dataset_metrics(eda_results, metrics=None, output_dir="plots", filename="dataset_metrics.png"):
    """
    Plot metrics across datasets
    
    Args:
        eda_results (list): List of EDA result dictionaries
        metrics (list): List of metrics to plot (default: brightness, contrast, saturation, edge_density)
        output_dir (str): Directory to save plots
        filename (str): Filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    print("Generating dataset metrics plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Default metrics if none provided
    if metrics is None:
        metrics = [
            ('avg_brightness', 'Brightness'),
            ('avg_contrast', 'Contrast'),
            ('avg_saturation', 'Saturation'),
            ('avg_edge_density', 'Edge Density')
        ]
    
    # Extract data
    data = []
    for eda in eda_results:
        dataset_name = eda['dataset_name']
        for metric_key, metric_name in metrics:
            if metric_key in eda:
                data.append({
                    'Dataset': dataset_name,
                    'Metric': metric_name,
                    'Value': eda[metric_key]
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Dataset', y='Value', hue='Metric', data=df)
    plt.title('Metrics Across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"  Dataset metrics plot saved to {output_path}")
    return output_path

def plot_dataset_comparison(original_eda, processed_eda, metric, output_dir="plots", filename=None):
    """
    Plot comparison between original and processed datasets for a specific metric
    
    Args:
        original_eda (list): List of original EDA result dictionaries
        processed_eda (list): List of processed EDA result dictionaries
        metric (tuple): Tuple (metric_key, metric_name) to plot
        output_dir (str): Directory to save plots
        filename (str): Optional filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    print(f"Generating comparison plot for {metric[1]}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    metric_key, metric_name = metric
    data = []
    
    # Process original data
    for eda in original_eda:
        if metric_key in eda:
            data.append({
                'Dataset': eda['dataset_name'],
                'Type': 'Original',
                'Value': eda[metric_key]
            })
    
    # Process processed data
    for eda in processed_eda:
        if metric_key in eda:
            # Remove suffix from dataset name if present
            dataset_name = eda['dataset_name']
            for suffix in ['_dedup', '_augmented']:
                if dataset_name.endswith(suffix):
                    dataset_name = dataset_name[:-len(suffix)]
                    break
            
            # Add processed type label
            process_type = 'Original'
            if eda['dataset_name'].endswith('_dedup'):
                process_type = 'Deduplicated'
            elif eda['dataset_name'].endswith('_augmented'):
                process_type = 'Augmented'
                
            data.append({
                'Dataset': dataset_name,
                'Type': process_type,
                'Value': eda[metric_key]
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # If filename not provided, create one based on metric
    if filename is None:
        filename = f"{metric_key}_comparison.png"
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Dataset', y='Value', hue='Type', data=df)
    plt.title(f'{metric_name} Comparison Across Processing Stages')
    plt.xlabel('Dataset')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.legend(title='Process Stage')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"  Comparison plot saved to {output_path}")
    return output_path

def plot_duplicate_summary(original_counts, dedup_counts, output_dir="plots", filename="duplicate_summary.png"):
    """
    Plot summary of duplicate removal
    
    Args:
        original_counts (dict): Dictionary mapping dataset names to original counts
        dedup_counts (dict): Dictionary mapping dataset names to duplicate counts
        output_dir (str): Directory to save plots
        filename (str): Filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    print("Generating duplicate removal summary plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    labels = list(original_counts.keys())
    removed = [dedup_counts.get(name, 0) for name in labels]
    kept = [original_counts.get(name, 0) - dedup_counts.get(name, 0) for name in labels]
    
    # Plot
    plt.figure(figsize=(12, 8))
    width = 0.35
    x = np.arange(len(labels))
    
    plt.bar(x, kept, width, label='Unique Images')
    plt.bar(x, removed, width, bottom=kept, label='Duplicates')
    
    plt.ylabel('Number of Images')
    plt.title('Duplicate Removal Summary')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    
    # Add percentages
    for i, (k, r) in enumerate(zip(kept, removed)):
        total = k + r
        if total > 0:
            duplicate_percent = 100 * r / total
            plt.text(i, total + 5, f"{duplicate_percent:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"  Duplicate removal summary plot saved to {output_path}")
    return output_path

def plot_augmentation_summary(original_counts, augmented_counts, output_dir="plots", filename="augmentation_summary.png"):
    """
    Plot summary of data augmentation
    
    Args:
        original_counts (dict): Dictionary mapping dataset names to original counts
        augmented_counts (dict): Dictionary mapping dataset names to augmented total counts
        output_dir (str): Directory to save plots
        filename (str): Filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    print("Generating augmentation summary plot...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    labels = list(original_counts.keys())
    original = [original_counts.get(name, 0) for name in labels]
    augmented = [augmented_counts.get(name, 0) - original_counts.get(name, 0) for name in labels]
    
    # Plot
    plt.figure(figsize=(12, 8))
    width = 0.35
    x = np.arange(len(labels))
    
    plt.bar(x, original, width, label='Original Images')
    plt.bar(x, augmented, width, bottom=original, label='Augmented Images')
    
    plt.ylabel('Number of Images')
    plt.title('Data Augmentation Summary')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    
    # Add augmentation factors
    for i, (o, a) in enumerate(zip(original, augmented)):
        if o > 0:
            aug_factor = (o + a) / o
            plt.text(i, o + a + 5, f"{aug_factor:.1f}x", ha='center')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"  Augmentation summary plot saved to {output_path}")
    return output_path

def plot_pipeline_flow(original_counts, dedup_counts, augmented_counts, output_dir="plots", filename="pipeline_flow.png"):
    """
    Plot the flow of images through the preprocessing pipeline
    
    Args:
        original_counts (dict): Dictionary mapping dataset names to original counts
        dedup_counts (dict): Dictionary mapping dataset names to duplicate removal counts
        augmented_counts (dict): Dictionary mapping dataset names to augmented total counts
        output_dir (str): Directory to save plots
        filename (str): Filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    print("Generating pipeline flow visualization...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process datasets in the same order
    datasets = list(original_counts.keys())
    
    # Create figure
    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]  # Make it iterable for single dataset case
    
    # Set common y-label
    fig.text(0.04, 0.5, 'Number of Images', va='center', rotation='vertical')
    
    # Plot each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Get counts
        original = original_counts.get(dataset, 0)
        duplicates = dedup_counts.get(dataset, 0)
        unique = original - duplicates
        augmented = augmented_counts.get(dataset, 0)
        added = augmented - unique
        
        # Plot the flow
        stages = ['Original', 'After\nDeduplication', 'After\nAugmentation']
        counts = [original, unique, augmented]
        
        ax.plot(stages, counts, 'o-', linewidth=2, markersize=10)
        
        # Add count values
        for j, count in enumerate(counts):
            ax.text(j, count + (max(counts) * 0.02), f"{count}", ha='center')
        
        # Annotate transitions
        if duplicates > 0:
            ax.annotate(f"-{duplicates}\nduplicates", 
                       xy=(0.5, (original + unique) / 2), 
                       xytext=(0.3, (original + unique) / 2 + (max(counts) * 0.1)), 
                       arrowprops=dict(arrowstyle='->'))
        
        if added > 0:
            ax.annotate(f"+{added}\naugmented", 
                       xy=(1.5, (unique + augmented) / 2), 
                       xytext=(1.7, (unique + augmented) / 2 + (max(counts) * 0.1)), 
                       arrowprops=dict(arrowstyle='->'))
        
        # Set title and adjust layout
        ax.set_title(dataset)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set overall title
    plt.suptitle('Dataset Flow Through Preprocessing Pipeline', fontsize=16)
    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"  Pipeline flow visualization saved to {output_path}")
    return output_path

def create_visual_report(original_eda, dedup_eda, aug_eda, dedup_counts, aug_counts, output_dir="plots"):
    """
    Create a comprehensive visual report of the preprocessing pipeline
    
    Args:
        original_eda (list): List of original EDA result dictionaries
        dedup_eda (list): List of deduplicated EDA result dictionaries
        aug_eda (list): List of augmented EDA result dictionaries
        dedup_counts (dict): Dictionary mapping dataset names to duplicate counts
        aug_counts (dict): Dictionary mapping dataset names to augmented total counts
        output_dir (str): Directory to save plots
        
    Returns:
        list: List of paths to saved plots
    """
    print("\nCreating visual report of preprocessing pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = []
    
    # Extract original counts
    original_counts = {eda['dataset_name']: eda['total_images'] for eda in original_eda}
    
    # Extract total counts after augmentation
    augmented_total_counts = {
        eda['dataset_name'].replace('_augmented', ''): eda['total_images'] 
        for eda in aug_eda if 'total_images' in eda
    }
    
    # Common metrics to plot
    metrics = [
        ('avg_brightness', 'Brightness'),
        ('avg_contrast', 'Contrast'),
        ('avg_saturation', 'Saturation'),
        ('avg_edge_density', 'Edge Density')
    ]
    
    # Generate plots
    
    # 1. Pipeline flow visualization
    plot_paths.append(
        plot_pipeline_flow(original_counts, dedup_counts, augmented_total_counts, output_dir)
    )
    
    # 2. Duplicate removal summary
    plot_paths.append(
        plot_duplicate_summary(original_counts, dedup_counts, output_dir)
    )
    
    # 3. Augmentation summary
    plot_paths.append(
        plot_augmentation_summary(
            {eda['dataset_name'].replace('_dedup', ''): eda['total_images'] for eda in dedup_eda},
            augmented_total_counts, 
            output_dir
        )
    )
    
    # 4. Metric comparisons
    for metric in metrics:
        plot_paths.append(
            plot_dataset_comparison(
                original_eda, 
                dedup_eda + aug_eda, 
                metric, 
                output_dir
            )
        )
    
    print(f"\nVisual report created with {len(plot_paths)} plots saved to {output_dir}")
    return plot_paths

def run_visualization_pipeline(original_eda, dedup_eda, aug_eda, dedup_counts, aug_stats, output_dir="plots"):
    """
    Run the full visualization pipeline
    
    Args:
        original_eda (list): List of original EDA result dictionaries
        dedup_eda (list): List of deduplicated EDA result dictionaries
        aug_eda (list): List of augmented EDA result dictionaries
        dedup_counts (dict): Dictionary mapping dataset names to duplicate counts
        aug_stats (dict): Dictionary mapping dataset names to augmentation stats
        output_dir (str): Directory to save plots
        
    Returns:
        list: List of paths to saved plots
    """
    print("Starting visualization pipeline...")
    
    # Extract augmented counts from stats
    aug_counts = {
        name: stats['total_images'] for name, stats in aug_stats.items()
    }
    
    # Create comprehensive visual report
    plot_paths = create_visual_report(
        original_eda,
        dedup_eda,
        aug_eda,
        dedup_counts,
        aug_counts,
        output_dir
    )
    
    print("Visualization pipeline complete!")
    return plot_paths

if __name__ == "__main__":
    # This script can be run standalone, but is intended to be used as part of a pipeline
    print("This module provides visualization functionality for sign language datasets.")
    print("Import and use the functions in your pipeline or provide data to visualize.")
    
    # If run directly, try to load sample data if available
    try:
        excel_dir = "excel"
        if os.path.exists(excel_dir):
            # Try to load EDA data
            original_df = pd.read_excel(os.path.join(excel_dir, "standardized_datasets_eda.xlsx"))
            dedup_df = pd.read_excel(os.path.join(excel_dir, "deduplicated_datasets_eda.xlsx"))
            
            if os.path.exists(os.path.join(excel_dir, "augmentation_stats.xlsx")):
                aug_df = pd.read_excel(os.path.join(excel_dir, "augmentation_stats.xlsx"))
                print("\nFound EDA data files. You can use this module to visualize them.")
    except Exception as e:
        print(f"Error loading sample data: {e}")
        print("You'll need to provide your own data for visualization.") 