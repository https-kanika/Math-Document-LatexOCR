import os
import pandas as pd
import xml.etree.ElementTree as ET
import glob
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import shutil
from sklearn.model_selection import train_test_split
import random

def get_traces_data(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """Extract trace data from InkML file"""
    traces_data = []
    
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    
    # Get all traces
    for trace_tag in root.findall(f'{xmlns}trace'):
        trace_coords = []
        
        # Get coords as text
        coords_text = trace_tag.text.strip()
        for coord_text in coords_text.split(','):
            if coord_text.strip():
                coords = coord_text.strip().split()
                if len(coords) >= 2:  # Ensure we have at least x and y
                    x, y = float(coords[0]), float(coords[1])
                    trace_coords.append((x, y))
        
        traces_data.append(trace_coords)
    
    return traces_data

def convert_to_img(traces_data, width=800, height=240, is_symbol=False, quality='medium'):
    """
    Convert traces to images with configurable quality settings for math expressions
    Uses square dimensions for symbols and rectangular for regular expressions
    
    Args:
        traces_data: List of traces data
        width: Target width for regular expressions
        height: Target height for regular expressions
        is_symbol: Whether this is a symbol (uses square dimensions)
        quality: Quality level ('low', 'medium', 'high')
    
    Returns:
        The rendered image
    """
    # Quality settings
    if quality == 'low':
        scale_factor = 1.0  # No upsampling
        line_thickness_regular = 3
        line_thickness_symbol = 4
        apply_blur = False
        apply_sharpen = False
    elif quality == 'medium':
        scale_factor = 1.2  # Modest upsampling
        line_thickness_regular = 4
        line_thickness_symbol = 5
        apply_blur = True
        apply_sharpen = False
    else:  # high quality (original settings)
        scale_factor = 1.5
        line_thickness_regular = 6
        line_thickness_symbol = 7
        apply_blur = True
        apply_sharpen = True
    
    # For symbols, use square dimensions
    if is_symbol:
        width = 800  # Square size for symbols
        height = 800
    
    # Find the bounding box
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for trace in traces_data:
        for x, y in trace:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    # Create high-resolution image for rendering
    render_width = int(width * scale_factor)
    render_height = int(height * scale_factor)
    img = np.ones((render_height, render_width), dtype=np.uint8) * 255
    
    # Calculate scaling factor
    trace_width = max_x - min_x
    trace_height = max_y - min_y
    
    # Avoid division by zero
    if trace_width == 0: trace_width = 1
    if trace_height == 0: trace_height = 1
    
    # Determine scale to fit within image dimensions with padding
    padding_x = int(width * 0.05)  # 5% padding
    padding_y = int(height * 0.10)  # 10% padding
    
    # For symbols, use equal padding on all sides to maintain aspect ratio
    if is_symbol:
        padding_x = int(width * 0.10)  # 10% padding all around for symbols
        padding_y = int(height * 0.10)
    
    scale = min((render_width - 2*padding_x) / trace_width, (render_height - 2*padding_y) / trace_height)
    
    # Center the equation
    offset_x = (render_width - trace_width * scale) / 2
    offset_y = (render_height - trace_height * scale) / 2
    
    # Draw the traces with optimized rendering
    for trace in traces_data:
        # Skip traces with less than 2 points
        if len(trace) < 2:
            continue
            
        # Convert to numpy array for vectorized operations
        points = np.array([(int((x - min_x) * scale + offset_x), int((y - min_y) * scale + offset_y)) 
                           for x, y in trace])
        
        # Clip points to image boundaries
        points[:, 0] = np.clip(points[:, 0], 0, render_width-1)
        points[:, 1] = np.clip(points[:, 1], 0, render_height-1)
        
        # Draw lines with appropriate thickness based on quality settings
        line_thickness = line_thickness_symbol if is_symbol else line_thickness_regular
        for i in range(1, len(points)):
            cv2.line(img, tuple(points[i-1]), tuple(points[i]), 0, line_thickness, cv2.LINE_AA)
    
    # Apply slight Gaussian blur to smooth edges if enabled
    if apply_blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Resize back to target dimensions with appropriate interpolation
    if scale_factor != 1.0:
        interpolation = cv2.INTER_LANCZOS4 if quality == 'high' else cv2.INTER_AREA
        img = cv2.resize(img, (width, height), interpolation=interpolation)
    
    # Sharpen the image for clearer lines if enabled
    if apply_sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    
    return img

def inkml2img_robust(input_path, output_path, is_symbol=False, quality='medium', compression=9):
    """
    Robustly convert InkML to PNG images with configurable quality
    
    Args:
        input_path: Path to input InkML file
        output_path: Path to save output PNG
        is_symbol: Whether this is a symbol (uses square dimensions)
        quality: Quality level ('low', 'medium', 'high')
        compression: PNG compression level (0-9, where 9 is maximum compression)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        traces_data = get_traces_data(input_path)
        if not traces_data:
            print(f"No trace data found in {input_path}")
            return False
        
        img = convert_to_img(traces_data, is_symbol=is_symbol, quality=quality)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with compression settings
        cv2.imwrite(output_path, img, [
            cv2.IMWRITE_PNG_COMPRESSION, compression,
        ])
        return True
    except Exception as e:
        print(f"Error converting {input_path} to image: {str(e)}")
        return False



def extract_labels_from_inkml(inkml_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """
    Extract labels and annotations from InkML file
    Returns a dictionary with label, normalizedLabel, and other metadata
    """
    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        
        # Initialize empty dictionary to store annotations
        annotations = {
            "sample_id": os.path.basename(inkml_path).split('.')[0],
            "label": None,
            "normalized_label": None,
            "split": None,
            "ink_creation_method": None,
            "label_creation_method": None
        }
        
        # Extract annotations
        for annotation in root.findall(f"{xmlns}annotation"):
            key = annotation.get('type', '')
            value = annotation.text
            
            if key == "label":
                annotations["label"] = value
            elif key == "normalizedLabel":
                annotations["normalized_label"] = value
            elif key == "splitTagOriginal":
                annotations["split"] = value
            elif key == "inkCreationMethod":
                annotations["ink_creation_method"] = value
            elif key == "labelCreationMethod":
                annotations["label_creation_method"] = value
        
        return annotations
    except Exception as e:
        print(f"Error parsing {inkml_path}: {str(e)}")
        return {
            "sample_id": os.path.basename(inkml_path).split('.')[0],
            "label": None,
            "normalized_label": None,
            "split": None,
            "ink_creation_method": None,
            "label_creation_method": None
        }
    
def process_dataset(base_dir, output_dir, max_files=None, quality='medium', compression=9):
    """
    Process all InkML files in the dataset and create a CSV database.
    Keeps symbols in a separate directory while pooling and splitting other data.
    
    Args:
        base_dir: Base directory containing InkML files
        output_dir: Output directory for images and CSV
        max_files: Maximum number of files to process (for testing)
        quality: Quality level ('low', 'medium', 'high')
        compression: PNG compression level (0-9, where 9 is maximum compression)
    """
    print(f"Looking for InkML files in: {base_dir}")
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"ERROR: Input directory '{base_dir}' does not exist!")
        return pd.DataFrame()
    
    # Create temporary output directories with original split structure
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create output directories for pooled data and symbols
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "symbols"), exist_ok=True)  # Keep symbols separate
    
    # Create temp directories with original split structure
    for split in ["train", "test", "val", "synthetic", "symbols"]:
        os.makedirs(os.path.join(temp_dir, split), exist_ok=True)
    
    # Find all InkML files in subdirectories
    all_inkml_files = []
    for split in ["train", "test", "valid", "synthetic", "symbols"]:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            inkml_files = glob.glob(os.path.join(split_dir, "*.inkml"))
            print(f"Found {len(inkml_files)} InkML files in {split} directory")
            for file in inkml_files:
                # For the "valid" split, normalize to "val"
                normalized_split = "val" if split == "valid" else split
                all_inkml_files.append((file, normalized_split))
        else:
            print(f"Directory not found: {split_dir}")
    
    # Check if any files were found
    if not all_inkml_files:
        print("ERROR: No InkML files found in the specified directories!")
        print("Please check the directory structure and paths.")
        return pd.DataFrame()
    
    # Limit number of files if specified
    if max_files and max_files > 0:
        # Ensure we have a balanced sample if max_files is specified
        random.seed(42)  # For reproducibility
        
        # Shuffle the files first
        random.shuffle(all_inkml_files)
        
        # Take the first max_files
        all_inkml_files = all_inkml_files[:max_files]
    
    print(f"Processing {len(all_inkml_files)} InkML files...")
    print(f"Using quality level: {quality}, compression: {compression}")
    
    # Initialize list to store data for DataFrame
    data = []
    
    # Process each InkML file and save to temp directory
    for inkml_file, split in tqdm(all_inkml_files, desc="Converting InkML to PNG"):
        try:
            # Extract sample ID from filename
            sample_id = os.path.basename(inkml_file).split('.')[0]
            
            # Check if this is a symbol file (for square image generation)
            is_symbol = (split == "symbols")
                
            # Define output image path in the temporary folder
            png_path = os.path.join(temp_dir, split, f"{sample_id}.png")
            
            # Extract annotations before converting
            annotations = extract_labels_from_inkml(inkml_file)
            
            # Convert InkML to PNG using our robust function with quality settings
            conversion_success = inkml2img_robust(
                inkml_file, png_path, is_symbol=is_symbol, 
                quality=quality, compression=compression
            )
            
            if not conversion_success:
                print(f"Skipping {inkml_file} due to conversion error")
                continue
            
            # Add row to data with additional metadata
            data.append({
                "filename": f"{sample_id}.png",
                "sample_id": sample_id,
                "label": annotations["label"],
                "normalized_label": annotations["normalized_label"],
                "original_split": split,  # Keep track of the original split
                "ink_creation_method": annotations["ink_creation_method"],
                "label_creation_method": annotations["label_creation_method"],
                "original_path": inkml_file,
                "is_symbol": is_symbol,
                "temp_path": png_path  # Store the temporary path for redistribution
            })
            
        except Exception as e:
            print(f"Error processing {inkml_file}: {str(e)}")
    
    # Check if any data was processed
    if not data:
        print("No data was processed successfully!")
        return pd.DataFrame()
    
    # Create DataFrame from the processed data
    df = pd.DataFrame(data)
    
    # Separate symbols from other data
    symbols_df = df[df['is_symbol'] == True].copy()
    regular_df = df[df['is_symbol'] == False].copy()
    
    print(f"\nSeparating symbols ({len(symbols_df)} files) from regular expressions ({len(regular_df)} files)")
    
    # For symbols, keep them in their own directory
    print("\nCopying symbol images to symbols directory...")
    for _, row in tqdm(symbols_df.iterrows(), total=len(symbols_df), desc="Processing symbols"):
        src_path = row['temp_path']
        dst_path = os.path.join(output_dir, "symbols", row['filename'])
        
        try:
            shutil.copy2(src_path, dst_path)
            # Mark that this is in the symbols directory for the CSV
            row['split'] = 'symbols'
        except Exception as e:
            print(f"Error copying symbol {src_path} to {dst_path}: {str(e)}")
    
    # For regular expressions, split them into train/val/test
    print(f"\nGenerating new train/val/test split with ratio 70-15-15 for {len(regular_df)} regular expressions...")
    
    # Split regular data into train, validation and test
    if len(regular_df) > 0:
        train_df, temp_df = train_test_split(regular_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Assign new split labels
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Combine dataframes for regular expressions
        regular_combined_df = pd.concat([train_df, val_df, test_df])
        
        # Print statistics about the new split
        print("\nNew dataset split statistics (regular expressions only):")
        split_stats = regular_combined_df['split'].value_counts()
        for split, count in split_stats.items():
            print(f"  {split}: {count} samples ({count/len(regular_combined_df)*100:.1f}%)")
        
        # Copy regular images to their new split directories
        print("\nCopying regular expression images to their new split directories...")
        for _, row in tqdm(regular_combined_df.iterrows(), total=len(regular_combined_df), desc="Redistributing regular expressions"):
            src_path = row['temp_path']
            dst_path = os.path.join(output_dir, row['split'], row['filename'])
            
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {str(e)}")
    else:
        print("No regular expressions to split.")
        regular_combined_df = pd.DataFrame()
    
    # Combine symbols and regular expressions dataframes
    combined_df = pd.concat([symbols_df, regular_combined_df]) if not regular_combined_df.empty else symbols_df
    
    # Shuffle rows for good measure
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Remove the temp directory
    try:
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not remove temporary directory {temp_dir}: {str(e)}")
    
    # Remove temporary path column before saving
    combined_df = combined_df.drop(columns=['temp_path'])
    
    # Save the final DataFrame to CSV
    csv_path = os.path.join(output_dir, "mathwriting_database.csv")
    combined_df.to_csv(csv_path, index=False)

    # Save dedicated symbols CSV
    symbols_csv_path = os.path.join(output_dir, "symbols_database.csv")
    symbols_df = combined_df[combined_df['is_symbol'] == True].copy()
    # Keep only essential columns for symbols
    symbols_df = symbols_df[['filename', 'sample_id', 'label', 'normalized_label']]
    symbols_df.to_csv(symbols_csv_path, index=False)
    print(f"Saved symbols CSV with {len(symbols_df)} entries to {symbols_csv_path}")

    # Also save split-specific CSVs
    for split_name in combined_df['split'].unique():
        split_df = combined_df[combined_df['split'] == split_name]
        split_csv = os.path.join(output_dir, f"{split_name}_database.csv")
        
        # Keep only essential columns and save
        split_df = split_df[['filename', 'sample_id', 'label', 'normalized_label', 'is_symbol']]
        split_df.to_csv(split_csv, index=False)
        print(f"Saved {split_name} CSV with {len(split_df)} entries")
    
    # Create a simplified version of the combined CSV with just essential columns
    simplified_df = combined_df[['filename', 'sample_id', 'label', 'normalized_label', 'split', 'is_symbol']]
    simplified_csv = os.path.join(output_dir, "mathwriting_database_simplified.csv")
    simplified_df.to_csv(simplified_csv, index=False)
    
    print(f"\nDatabase created with {len(combined_df)} entries")
    print(f"Full CSV saved to {csv_path}")
    print(f"Simplified CSV saved to {simplified_csv}")
    print(f"Symbol images are in {os.path.join(output_dir, 'symbols')}")
    print(f"Regular expression images are in {os.path.join(output_dir, '[train|val|test]')}")
    
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert InkML files to PNG and create database')
    parser.add_argument('--input', type=str, required=True, help='Base directory containing train/test/valid folders')
    parser.add_argument('--output', type=str, required=True, help='Output directory for images and CSV')
    parser.add_argument('--max', type=int, default=None, help='Maximum number of files to process (for testing)')
    parser.add_argument('--quality', type=str, default='medium', choices=['low', 'medium', 'high'],
                        help='Quality level for rendered images (low, medium, high)')
    parser.add_argument('--compression', type=int, default=6, choices=range(0, 10),
                        help='PNG compression level (0-9, where 9 is maximum compression)')
    
    args = parser.parse_args()
    
    # Fix potential command line parsing issues
    input_dir = args.input.rstrip('\\/"\'')
    output_dir = args.output.rstrip('\\/"\'')
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max files: {args.max}")
    print(f"Quality level: {args.quality}")
    print(f"Compression level: {args.compression}")
    
    df = process_dataset(input_dir, output_dir, args.max, args.quality, args.compression)
    
    if df.empty:
        print("Warning: No data was processed!")
    else:
        print(f"\nSuccessfully processed {len(df)} files.")
        print(f"Images saved in {output_dir}/[train|val|test] directories")
        print("Data pooled from all original folders and split into 70-15-15 train/val/test")