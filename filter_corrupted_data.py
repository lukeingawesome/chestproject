#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter corrupted or unloadable data from CSV and save a clean version.

This script will:
1. Load the original CSV
2. Check each image path for validity
3. Try to load each image to verify it's not corrupted
4. Save a new CSV with only valid data
5. Provide statistics about what was filtered out
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_image_validity(img_path, check_load=True):
    """
    Check if an image path is valid and optionally try to load it.
    
    Args:
        img_path (str): Path to the image file
        check_load (bool): Whether to actually try loading the image
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not img_path or pd.isna(img_path) or img_path == "-1":
        return False, "Empty or invalid path"
    
    # Check if file exists
    p = Path(img_path)
    if not p.exists():
        return False, f"File does not exist: {img_path}"
    
    # Check file size
    try:
        file_size = p.stat().st_size
        if file_size == 0:
            return False, f"File is empty: {img_path}"
    except Exception as e:
        return False, f"Cannot access file: {img_path} - {str(e)}"
    
    # Try to load the image if requested
    if check_load:
        try:
            # Handle different file types
            if img_path.lower().endswith('.npz'):
                # Handle numpy compressed files
                with np.load(img_path) as npz:
                    # Check if the file has any data
                    if len(npz.files) == 0:
                        return False, f"NPZ file has no data: {img_path}"
                    # Try to access the first array to verify it's not corrupted
                    first_key = list(npz.files)[0]
                    _ = npz[first_key]
            else:
                # Handle regular image files
                with Image.open(img_path) as img:
                    # Try to convert to RGB to verify it's a valid image
                    img.convert("RGB")
            
            return True, "OK"
            
        except EOFError as e:
            return False, f"EOFError - corrupted file: {img_path}"
        except Exception as e:
            return False, f"Failed to load image: {img_path} - {str(e)}"
    
    return True, "OK"

def filter_dataset(csv_path, output_path=None, check_load=True, sample_size=None):
    """
    Filter the dataset and remove corrupted/unloadable entries.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_path (str): Path to save the filtered CSV (optional)
        check_load (bool): Whether to actually try loading images
        sample_size (int): If provided, only process this many samples (for testing)
    
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    logger.info(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    original_count = len(df)
    logger.info(f"Original dataset size: {original_count}")
    
    # Determine image column name
    possible_img_columns = ['img_path', 'image_path', 'image', 'file_path', 'path']
    img_column = None
    for col in possible_img_columns:
        if col in df.columns:
            img_column = col
            break
    
    if img_column is None:
        logger.error("Could not find image path column. Available columns:")
        for col in df.columns:
            logger.error(f"  - {col}")
        raise ValueError("No image path column found")
    
    logger.info(f"Using image column: {img_column}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampling {sample_size} rows for testing")
    
    # Check each image
    valid_indices = []
    error_counts = {}
    
    logger.info("Checking image validity...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        img_path = row[img_column]
        is_valid, error_msg = check_image_validity(img_path, check_load)
        
        if is_valid:
            valid_indices.append(idx)
        else:
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
    
    # Create filtered dataframe
    filtered_df = df.iloc[valid_indices].reset_index(drop=True)
    
    # Print statistics
    logger.info(f"\n{'='*50}")
    logger.info("FILTERING STATISTICS")
    logger.info(f"{'='*50}")
    logger.info(f"Original samples: {original_count}")
    logger.info(f"Valid samples: {len(filtered_df)}")
    logger.info(f"Removed samples: {original_count - len(filtered_df)}")
    logger.info(f"Removal rate: {((original_count - len(filtered_df)) / original_count * 100):.2f}%")
    
    if error_counts:
        logger.info(f"\nError breakdown:")
        for error_msg, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {error_msg}: {count}")
    
    # Save filtered dataset
    if output_path:
        logger.info(f"\nSaving filtered dataset to: {output_path}")
        filtered_df.to_csv(output_path, index=False)
        logger.info("✅ Filtered dataset saved successfully!")
    
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Filter corrupted data from CSV")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output CSV file path (optional)")
    parser.add_argument("--no-load-check", action="store_true", 
                       help="Skip actual image loading (faster, less thorough)")
    parser.add_argument("--sample", "-s", type=int, 
                       help="Process only N samples (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set output path if not provided
    if not args.output:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"
    else:
        output_path = args.output
    
    try:
        filtered_df = filter_dataset(
            csv_path=args.input,
            output_path=output_path,
            check_load=not args.no_load_check,
            sample_size=args.sample
        )
        
        logger.info(f"\n🎉 Filtering completed successfully!")
        logger.info(f"Clean dataset saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during filtering: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 