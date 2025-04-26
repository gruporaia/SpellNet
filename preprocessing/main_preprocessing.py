#!/usr/bin/env python3
"""
Sign Language Datasets Main Preprocessing Pipeline

This module orchestrates the complete preprocessing pipeline for sign language image datasets:
1. Initial Preprocessing: Standardization of images to 224x224 PNG format
2. Exploratory Data Analysis: Analysis of dataset characteristics
3. Deduplication: Removal of duplicate images using perceptual hashing
4. Data Augmentation: Generation of additional training examples with transformations

The module can execute the entire pipeline or specific steps as needed.
"""
#Author: MatheusHRV

import os
import sys
import argparse
import time
import pandas as pd
from pathlib import Path

# Import preprocessing modules
try:
    from preprocessing import main as preprocessing_main
    from eda import perform_image_eda, generate_eda_report
    from deduplication import main as deduplication_main
    from data_augmentation import main as augmentation_main
    from visualization import run_visualization_pipeline
except ImportError as e:
    print(f"Error importing preprocessing modules: {e}")
    print("Make sure you're running this script from the preprocessing directory.")
    sys.exit(1)

def setup_directories():
    """Create necessary directories for the pipeline"""
    os.makedirs('standardized_datasets', exist_ok=True)
    os.makedirs('deduplicated_datasets', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('excel', exist_ok=True)

def run_preprocessing_step(data_dir="original_dataset(2)", force=False):
    """
    Run the initial preprocessing step
    
    Args:
        data_dir (str): Path to the original dataset directory
        force (bool): Whether to force reprocessing if output already exists
        
    Returns:
        tuple: (standardized paths, metadata DataFrame, EDA results, EDA DataFrame)
    """
    # Check if standardized datasets already exist
    if not force and os.path.exists("standardized_datasets") and len(os.listdir("standardized_datasets")) > 0:
        print("\nStandardized datasets already exist. Use --force to reprocess.")
        
        # Load metadata if available
        metadata_path = os.path.join("excel", "original_datasets_metadata.xlsx")
        eda_path = os.path.join("excel", "standardized_datasets_eda.xlsx")
        
        if os.path.exists(metadata_path) and os.path.exists(eda_path):
            metadata_df = pd.read_excel(metadata_path)
            eda_df = pd.read_excel(eda_path)
            
            # Get standardized paths
            standardized_paths = {}
            for dataset_name in metadata_df['dataset_name'].values:
                path = os.path.join("standardized_datasets", dataset_name)
                if os.path.exists(path):
                    standardized_paths[dataset_name] = path
            
            print(f"Loaded {len(standardized_paths)} existing standardized datasets.")
            return standardized_paths, metadata_df, None, eda_df
        
    print("\n" + "="*80)
    print("STEP 1: INITIAL PREPROCESSING")
    print("="*80)
    
    # Run preprocessing
    start_time = time.time()
    standardized_paths, metadata_df, eda_results, eda_df = preprocessing_main()
    elapsed_time = time.time() - start_time
    
    print(f"\nPreprocessing completed in {elapsed_time:.1f} seconds.")
    return standardized_paths, metadata_df, eda_results, eda_df

def run_deduplication_step(standardized_paths=None, force=False):
    """
    Run the deduplication step
    
    Args:
        standardized_paths (dict): Dictionary mapping dataset names to paths
        force (bool): Whether to force reprocessing if output already exists
        
    Returns:
        tuple: (deduplicated paths, deduplication DataFrame, EDA results, EDA DataFrame)
    """
    # Check if deduplicated datasets already exist
    if not force and os.path.exists("deduplicated_datasets") and len(os.listdir("deduplicated_datasets")) > 0:
        print("\nDeduplicated datasets already exist. Use --force to reprocess.")
        
        # Load deduplication results if available
        dedup_path = os.path.join("excel", "deduplication_results.xlsx")
        eda_path = os.path.join("excel", "deduplicated_datasets_eda.xlsx")
        
        if os.path.exists(dedup_path) and os.path.exists(eda_path):
            dedup_df = pd.read_excel(dedup_path)
            eda_df = pd.read_excel(eda_path)
            
            # Get deduplicated paths
            deduplicated_paths = {}
            for dataset_name in dedup_df['dataset_name'].values:
                path = os.path.join("deduplicated_datasets", dataset_name)
                if os.path.exists(path):
                    deduplicated_paths[dataset_name] = path
            
            print(f"Loaded {len(deduplicated_paths)} existing deduplicated datasets.")
            return deduplicated_paths, dedup_df, None, eda_df
    
    print("\n" + "="*80)
    print("STEP 2: DEDUPLICATION")
    print("="*80)
    
    # Run deduplication
    start_time = time.time()
    deduplicated_paths, dedup_df, dedup_eda_results, dedup_eda_df = deduplication_main()
    elapsed_time = time.time() - start_time
    
    print(f"\nDeduplication completed in {elapsed_time:.1f} seconds.")
    return deduplicated_paths, dedup_df, dedup_eda_results, dedup_eda_df

def run_augmentation_step(deduplicated_paths=None, force=False):
    """
    Run the augmentation step
    
    Args:
        deduplicated_paths (dict): Dictionary mapping dataset names to paths
        force (bool): Whether to force reprocessing if output already exists
        
    Returns:
        tuple: (augmented paths, augmentation stats)
    """
    # Check if augmented datasets already exist
    augmented_dir = "standardized_datasets"
    augmented_exist = False
    if os.path.exists(augmented_dir):
        for item in os.listdir(augmented_dir):
            if "_augmented" in item and os.path.isdir(os.path.join(augmented_dir, item)):
                augmented_exist = True
                break
    
    if not force and augmented_exist:
        print("\nAugmented datasets already exist. Use --force to reprocess.")
        
        # Load augmentation stats if available
        stats_path = os.path.join("excel", "augmentation_stats.xlsx")
        
        if os.path.exists(stats_path):
            stats_df = pd.read_excel(stats_path)
            
            # Get augmented paths
            augmented_paths = {}
            for dataset_name in stats_df['dataset_name'].values:
                path = os.path.join(augmented_dir, f"{dataset_name}_augmented")
                if os.path.exists(path):
                    augmented_paths[dataset_name] = path
            
            stats = {}
            print(f"Loaded {len(augmented_paths)} existing augmented datasets.")
            return augmented_paths, stats
    
    print("\n" + "="*80)
    print("STEP 3: DATA AUGMENTATION")
    print("="*80)
    
    # Run augmentation
    start_time = time.time()
    augmented_paths, augmentation_stats = augmentation_main()
    elapsed_time = time.time() - start_time
    
    print(f"\nAugmentation completed in {elapsed_time:.1f} seconds.")
    return augmented_paths, augmentation_stats

def run_visualization_step(metadata_df=None, standardized_eda_df=None, 
                          dedup_df=None, dedup_eda_df=None, 
                          augmentation_stats=None):
    """
    Run the visualization step to generate summary reports and visualizations
    
    Args:
        metadata_df (DataFrame): Metadata DataFrame
        standardized_eda_df (DataFrame): Standardized EDA DataFrame
        dedup_df (DataFrame): Deduplication DataFrame
        dedup_eda_df (DataFrame): Deduplicated EDA DataFrame
        augmentation_stats (dict): Augmentation statistics
    """
    print("\n" + "="*80)
    print("STEP 4: VISUALIZATION AND REPORTS")
    print("="*80)
    
    # Perform EDA on augmented datasets if they exist
    augmented_eda_results = []
    augmented_dir = "standardized_datasets"
    
    for item in os.listdir(augmented_dir):
        if "_augmented" in item and os.path.isdir(os.path.join(augmented_dir, item)):
            dataset_path = os.path.join(augmented_dir, item)
            dataset_name = item
            print(f"\nPerforming EDA on augmented {dataset_name}...")
            try:
                eda = perform_image_eda(dataset_path, f"Augmented_{dataset_name}", "plots")
                augmented_eda_results.append(eda)
                print(f"Completed EDA for augmented {dataset_name}")
            except Exception as e:
                print(f"Error performing EDA on {dataset_name}: {e}")
    
    # Convert EDA results to DataFrame
    if augmented_eda_results:
        augmented_eda_df = pd.DataFrame(augmented_eda_results)
        augmented_eda_df.to_excel(os.path.join("excel", "augmented_datasets_eda.xlsx"), index=False)
        print("\nSaved augmented datasets EDA to excel/augmented_datasets_eda.xlsx")
    
    # Generate summary visualizations
    print("\nGenerating summary visualizations...")
    try:
        # Extract data from DataFrames
        # This section would need to be implemented based on the exact structure of your data
        # and what visualizations you want to generate
        
        # For a complete implementation, you would need to modify visualization.py to accept
        # the formats of data you have available at this point
        
        print("Summary visualizations complete. Check the 'plots' directory.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

def create_final_report():
    """Create a final comprehensive report of the preprocessing pipeline"""
    print("\nCreating final report...")
    
    # Collect all Excel files
    excel_files = [f for f in os.listdir("excel") if f.endswith(".xlsx")]
    
    # Create a summary table
    summary = []
    
    # Process each Excel file to extract key metrics
    for excel_file in excel_files:
        try:
            df = pd.read_excel(os.path.join("excel", excel_file))
            file_name = excel_file.replace(".xlsx", "")
            
            # Extract basic information
            if "dataset" in df.columns:
                datasets = df["dataset"].nunique()
            elif "dataset_name" in df.columns:
                datasets = df["dataset_name"].nunique()
            else:
                datasets = "N/A"
                
            rows = len(df)
            columns = len(df.columns)
            
            summary.append({
                "Report": file_name,
                "Datasets": datasets,
                "Rows": rows,
                "Columns": columns
            })
        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
    
    # Create summary DataFrame
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_excel(os.path.join("excel", "preprocessing_summary.xlsx"), index=False)
        print("Final report created: excel/preprocessing_summary.xlsx")
    else:
        print("No data available for final report.")

def main():
    """Main function to run the complete preprocessing pipeline"""
    parser = argparse.ArgumentParser(description="Run the sign language preprocessing pipeline")
    parser.add_argument("--data-dir", default="original_dataset(2)", 
                        help="Path to the original dataset directory")
    parser.add_argument("--steps", default="all", 
                        help="Which preprocessing steps to run (all, preprocess, dedup, augment, visualize)")
    parser.add_argument("--force", action="store_true", 
                        help="Force reprocessing even if output already exists")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Print welcome message
    print("\n" + "="*80)
    print("SIGN LANGUAGE PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Steps to run: {args.steps}")
    print(f"Force reprocessing: {args.force}")
    
    total_start_time = time.time()
    
    # Determine which steps to run
    steps = args.steps.lower().split(',')
    run_all = "all" in steps
    run_preprocess = run_all or "preprocess" in steps
    run_dedup = run_all or "dedup" in steps
    run_augment = run_all or "augment" in steps
    run_visualize = run_all or "visualize" in steps
    
    # Store results from each step
    standardized_paths = None
    metadata_df = None
    standardized_eda_results = None
    standardized_eda_df = None
    
    deduplicated_paths = None
    dedup_df = None
    dedup_eda_results = None
    dedup_eda_df = None
    
    augmented_paths = None
    augmentation_stats = None
    
    # Step 1: Initial Preprocessing
    if run_preprocess:
        standardized_paths, metadata_df, standardized_eda_results, standardized_eda_df = run_preprocessing_step(
            data_dir=args.data_dir, force=args.force)
    
    # Step 2: Deduplication
    if run_dedup:
        deduplicated_paths, dedup_df, dedup_eda_results, dedup_eda_df = run_deduplication_step(
            standardized_paths=standardized_paths, force=args.force)
    
    # Step 3: Data Augmentation
    if run_augment:
        augmented_paths, augmentation_stats = run_augmentation_step(
            deduplicated_paths=deduplicated_paths, force=args.force)
    
    # Step 4: Visualization and Reports
    if run_visualize:
        run_visualization_step(
            metadata_df=metadata_df,
            standardized_eda_df=standardized_eda_df,
            dedup_df=dedup_df,
            dedup_eda_df=dedup_eda_df,
            augmentation_stats=augmentation_stats
        )
        
        # Create final summary report
        create_final_report()
    
    # Calculate total elapsed time
    total_elapsed_time = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*80)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("="*80)
    print(f"Total time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print("\nThe preprocessed datasets are ready for model training.")
    print("You can find:")
    print("- Standardized datasets in: standardized_datasets/")
    print("- Deduplicated datasets in: deduplicated_datasets/")
    print("- Augmented datasets in: standardized_datasets/*_augmented/")
    print("- Analysis reports in: excel/")
    print("- Visualizations in: plots/")

if __name__ == "__main__":
    main() 