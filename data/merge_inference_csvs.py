#!/usr/bin/env python3
"""
Script to merge CSV files from inference_encode_results, shuffle, and split into train/valid sets.
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import random
from tqdm import tqdm


def find_csv_files(base_dir):
    """Find all CSV files in subdirectories of base_dir."""
    csv_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Directory {base_dir} does not exist")
    
    # Search for CSV files in all subdirectories
    for csv_file in base_path.rglob("*.csv"):
        csv_files.append(csv_file)
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  {csv_file}")
    
    return csv_files


def merge_csv_files(csv_files):
    """Merge all CSV files into a single DataFrame."""
    all_dataframes = []
    
    print("Merging CSV files...")
    for csv_file in tqdm(csv_files):
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            # Add source file information for tracking
            df['source_file'] = str(csv_file)
            all_dataframes.append(df)
            # print(f"  Loaded {len(df)} rows from {csv_file}")
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
    
    if not all_dataframes:
        raise ValueError("No valid CSV files found or all files failed to load")
    
    # Concatenate all dataframes
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total merged rows: {len(merged_df)}")
    
    return merged_df


def shuffle_and_split(df, num_valid_samples, random_seed=42):
    """Shuffle the dataframe and split into train/valid sets."""
    print(f"Shuffling data with random seed: {random_seed}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle the dataframe
    shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Check if we have enough samples for validation
    if num_valid_samples >= len(shuffled_df):
        raise ValueError(f"Number of validation samples ({num_valid_samples}) "
                        f"must be less than total samples ({len(shuffled_df)})")
    
    # Split into train and valid
    valid_df = shuffled_df.iloc[:num_valid_samples].copy()
    train_df = shuffled_df.iloc[num_valid_samples:].copy()
    
    print(f"Split results:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(valid_df)}")
    
    return train_df, valid_df


def save_splits(train_df, valid_df, output_dir, drop_source_file=True):
    """Save train and valid dataframes to specified directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Optionally remove source_file column before saving
    if drop_source_file:
        if 'source_file' in train_df.columns:
            train_df = train_df.drop('source_file', axis=1)
        if 'source_file' in valid_df.columns:
            valid_df = valid_df.drop('source_file', axis=1)
    
    # Save files
    train_path = output_path / "train.csv"
    valid_path = output_path / "valid.csv"
    
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    
    print(f"Saved files:")
    print(f"  Training data: {train_path} ({len(train_df)} rows)")
    print(f"  Validation data: {valid_path} ({len(valid_df)} rows)")
    
    return train_path, valid_path


def main():
    parser = argparse.ArgumentParser(description="Merge CSV files and split into train/valid sets")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="inference_encode_results",
        help="Directory containing CSV files to merge (default: inference_encode_results)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory to save train.csv and valid.csv"
    )
    parser.add_argument(
        "--num_valid", 
        type=int, 
        required=True,
        help="Number of samples to use for validation"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--keep_source_info", 
        action="store_true",
        help="Keep source file information in output CSVs"
    )
    
    args = parser.parse_args()
    
    try:
        # Find all CSV files
        csv_files = find_csv_files(args.input_dir)
        
        if not csv_files:
            print(f"No CSV files found in {args.input_dir}")
            return
        
        # Merge CSV files
        merged_df = merge_csv_files(csv_files)
        
        # Shuffle and split
        train_df, valid_df = shuffle_and_split(
            merged_df, 
            args.num_valid, 
            args.random_seed
        )
        
        # Save splits
        train_path, valid_path = save_splits(
            train_df, 
            valid_df, 
            args.output_dir,
            drop_source_file=not args.keep_source_info
        )
        
        print("\nProcess completed successfully!")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"  Input directory: {args.input_dir}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Total samples processed: {len(merged_df)}")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Validation samples: {len(valid_df)}")
        print(f"  Random seed used: {args.random_seed}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
