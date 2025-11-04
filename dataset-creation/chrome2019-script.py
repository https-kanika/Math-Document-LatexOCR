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


def parse_label_file(label_file_path):
    """
    Parse TXT file containing file paths and LaTeX labels
    
    Args:
        label_file_path: Path to the TXT file
    
    Returns:
        Dictionary mapping relative InkML paths to LaTeX labels
    """
    labels_dict = {}
    
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab to separate path and label
                parts = line.split('\t')
                if len(parts) >= 2:
                    inkml_path = parts[0]
                    latex_label = parts[1]
                    labels_dict[inkml_path] = latex_label
                else:
                    print(f"Warning: Skipping malformed line: {line}")
    
    except Exception as e:
        print(f"Error reading label file {label_file_path}: {str(e)}")
    
    return labels_dict



def convert_to_img_dynamic(traces_data, target_height=240, min_width=200, max_width=1200, padding_percent=0.15, quality='medium'):
    """
    Convert traces to images with DYNAMIC width based on actual content
    Height is fixed, width adjusts to maintain aspect ratio
    
    Args:
        traces_data: List of traces data
        target_height: Target height for images (default: 240)
        min_width: Minimum allowed width (default: 200)
        max_width: Maximum allowed width (default: 1200)
        padding_percent: Padding as percentage of dimension (default: 0.15 = 15%)
        quality: Quality level ('low', 'medium', 'high')
    
    Returns:
        The rendered image with dynamic width
    """
    # Quality settings
    if quality == 'low':
        scale_factor = 1.0
        line_thickness = 3
        apply_blur = False
        apply_sharpen = False
    elif quality == 'medium':
        scale_factor = 1.2
        line_thickness = 4
        apply_blur = True
        apply_sharpen = False
    else:  # high
        scale_factor = 1.5
        line_thickness = 6
        apply_blur = True
        apply_sharpen = True
    
    # Find the bounding box from actual traces
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for trace in traces_data:
        for x, y in trace:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    # Calculate actual content dimensions from InkML coordinates
    content_width = max_x - min_x
    content_height = max_y - min_y
    
    # Avoid division by zero
    if content_width == 0: content_width = 1
    if content_height == 0: content_height = 1
    
    # Calculate aspect ratio of the actual handwritten content
    aspect_ratio = content_width / content_height
    
    # Calculate output dimensions
    height = target_height
    width = int(target_height * aspect_ratio)
    
    # Clamp width to reasonable bounds
    width = max(min_width, min(width, max_width))
    
    # Recalculate height if we hit max_width (to maintain aspect ratio)
    if width == max_width and aspect_ratio > (max_width / target_height):
        height = int(max_width / aspect_ratio)
    
    # Calculate padding
    padding_x = int(width * padding_percent)
    padding_y = int(height * padding_percent)
    
    # Create high-resolution image for rendering
    render_width = int(width * scale_factor)
    render_height = int(height * scale_factor)
    render_padding_x = int(padding_x * scale_factor)
    render_padding_y = int(padding_y * scale_factor)
    
    img = np.ones((render_height, render_width), dtype=np.uint8) * 255
    
    # Calculate scale to fit content with padding
    scale_x = (render_width - 2 * render_padding_x) / content_width
    scale_y = (render_height - 2 * render_padding_y) / content_height
    scale = min(scale_x, scale_y)
    
    # Center the equation
    offset_x = (render_width - content_width * scale) / 2
    offset_y = (render_height - content_height * scale) / 2
    
    # Draw the traces
    for trace in traces_data:
        if len(trace) < 2:
            continue
            
        points = np.array([
            (int((x - min_x) * scale + offset_x), 
             int((y - min_y) * scale + offset_y)) 
            for x, y in trace
        ])
        
        # Clip points to image boundaries
        points[:, 0] = np.clip(points[:, 0], 0, render_width-1)
        points[:, 1] = np.clip(points[:, 1], 0, render_height-1)
        
        # Draw lines
        for i in range(1, len(points)):
            cv2.line(img, tuple(points[i-1]), tuple(points[i]), 0, line_thickness, cv2.LINE_AA)
    
    # Apply blur if enabled
    if apply_blur:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Resize back to target dimensions
    if scale_factor != 1.0:
        interpolation = cv2.INTER_LANCZOS4 if quality == 'high' else cv2.INTER_AREA
        img = cv2.resize(img, (width, height), interpolation=interpolation)
    
    # Sharpen if enabled
    if apply_sharpen:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    
    return img, width, height

def inkml2img_robust(input_path, output_path, is_symbol=False, quality='medium', compression=9, use_dynamic_width=True):
    """
    Robustly convert InkML to PNG images with configurable quality
    
    Args:
        input_path: Path to input InkML file
        output_path: Path to save output PNG
        is_symbol: Whether this is a symbol (uses square dimensions)
        quality: Quality level ('low', 'medium', 'high')
        compression: PNG compression level (0-9, where 9 is maximum compression)
        use_dynamic_width: If True, use dynamic width based on content
    
    Returns:
        Tuple of (success: bool, width: int, height: int)
    """
    try:
        traces_data = get_traces_data(input_path)
        if not traces_data:
            print(f"No trace data found in {input_path}")
            return False, 0, 0
        
        if use_dynamic_width:
            img, width, height = convert_to_img_dynamic(traces_data, quality=quality)
        else:
            img = convert_to_img(traces_data, is_symbol=is_symbol, quality=quality)
            height, width = img.shape
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with compression settings
        cv2.imwrite(output_path, img, [
            cv2.IMWRITE_PNG_COMPRESSION, compression,
        ])
        return True, width, height
    except Exception as e:
        print(f"Error converting {input_path} to image: {str(e)}")
        return False, 0, 0


def process_crohme2019_dataset(base_dir, output_dir, max_files=None, quality='medium', compression=9, use_dynamic_width=True):
    """
    Process CROHME2019 dataset with TXT label files
    
    Args:
        base_dir: Base directory containing crohme2019 folder and TXT files
        output_dir: Output directory for images and CSV
        max_files: Maximum number of files to process (for testing)
        quality: Quality level ('low', 'medium', 'high')
        compression: PNG compression level (0-9, where 9 is maximum compression)
        use_dynamic_width: If True, use dynamic width based on content aspect ratio
    """
    print(f"Processing CROHME2019 dataset from: {base_dir}")
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"ERROR: Input directory '{base_dir}' does not exist!")
        return pd.DataFrame()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Parse label files for each split
    split_file_mapping = {
        'train': 'crohme2019_train.txt',
        'val': 'crohme2019_valid.txt',
        'test': 'crohme2019_test.txt'
    }
    
    all_labels = {}
    
    print("\nParsing label files...")
    for split, filename in split_file_mapping.items():
        label_file = os.path.join(base_dir, filename)
        if os.path.exists(label_file):
            labels = parse_label_file(label_file)
            all_labels[split] = labels
            print(f"Found {len(labels)} labels in {split} file ({filename})")
        else:
            print(f"Warning: Label file not found: {label_file}")
            all_labels[split] = {}
    
    # Collect all files to process
    all_files = []
    for split in ['train', 'val', 'test']:
        for inkml_rel_path, latex_label in all_labels[split].items():
            inkml_abs_path = os.path.join(base_dir, inkml_rel_path)
            
            if os.path.exists(inkml_abs_path):
                all_files.append({
                    'abs_path': inkml_abs_path,
                    'rel_path': inkml_rel_path,
                    'split': split,
                    'label': latex_label
                })
            else:
                print(f"Warning: InkML file not found: {inkml_abs_path}")
    
    if not all_files:
        print("ERROR: No valid InkML files found!")
        return pd.DataFrame()
    
    # Limit number of files if specified
    if max_files and max_files > 0:
        random.seed(42)
        random.shuffle(all_files)
        all_files = all_files[:max_files]
    
    sizing_mode = "dynamic width" if use_dynamic_width else "fixed 800x240"
    print(f"\nProcessing {len(all_files)} InkML files...")
    print(f"Using quality level: {quality}, compression: {compression}, sizing: {sizing_mode}")
    
    # Initialize list to store data for DataFrame
    data = []
    
    # Track image dimensions for statistics
    widths = []
    heights = []
    
    # Process each file
    for file_info in tqdm(all_files, desc="Converting InkML to PNG"):
        try:
            sample_id = os.path.basename(file_info['abs_path']).split('.')[0]
            split = file_info['split']
            
            png_filename = f"{sample_id}.png"
            png_path = os.path.join(output_dir, split, png_filename)
            
            # Convert InkML to PNG with dynamic sizing
            conversion_success, img_width, img_height = inkml2img_robust(
                file_info['abs_path'], png_path, is_symbol=False, 
                quality=quality, compression=compression, use_dynamic_width=use_dynamic_width
            )
            
            if not conversion_success:
                print(f"Skipping {file_info['abs_path']} due to conversion error")
                continue
            
            # Track dimensions
            widths.append(img_width)
            heights.append(img_height)
            
            # Add row to data
            data.append({
                "filename": png_filename,
                "sample_id": sample_id,
                "label": file_info['label'],
                "split": split,
                "original_path": file_info['rel_path'],
                "image_width": img_width,
                "image_height": img_height
            })
            
        except Exception as e:
            print(f"Error processing {file_info['abs_path']}: {str(e)}")
    
    # Check if any data was processed
    if not data:
        print("No data was processed successfully!")
        return pd.DataFrame()
    
    # Create DataFrame from the processed data
    df = pd.DataFrame(data)
    
    # Print statistics
    print("\nDataset statistics:")
    split_stats = df['split'].value_counts()
    for split, count in split_stats.items():
        print(f"  {split}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Print image dimension statistics
    if widths and heights:
        print(f"\nImage dimension statistics:")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={int(np.mean(widths))}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={int(np.mean(heights))}")
    
    # Save the main DataFrame to CSV
    csv_path = os.path.join(output_dir, "crohme2019_database.csv")
    df.to_csv(csv_path, index=False)
    
    # Save split-specific CSVs
    for split_name in df['split'].unique():
        split_df = df[df['split'] == split_name]
        split_csv = os.path.join(output_dir, f"{split_name}_database.csv")
        split_df.to_csv(split_csv, index=False)
        print(f"Saved {split_name} CSV with {len(split_df)} entries")
    
    # Create a simplified version with essential columns
    simplified_df = df[['filename', 'sample_id', 'label', 'split', 'image_width', 'image_height']]
    simplified_csv = os.path.join(output_dir, "crohme2019_database_simplified.csv")
    simplified_df.to_csv(simplified_csv, index=False)
    
    print(f"\nDatabase created with {len(df)} entries")
    print(f"Full CSV saved to {csv_path}")
    print(f"Simplified CSV saved to {simplified_csv}")
    print(f"Images are in {os.path.join(output_dir, '[train|val|test]')}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CROHME2019 InkML files to PNG and create database')
    parser.add_argument('--input', type=str, required=True, 
                        help='Base directory containing crohme2019 folder and label TXT files')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output directory for images and CSV')
    parser.add_argument('--max', type=int, default=None, 
                        help='Maximum number of files to process (for testing)')
    parser.add_argument('--quality', type=str, default='medium', choices=['low', 'medium', 'high'],
                        help='Quality level for rendered images (low, medium, high)')
    parser.add_argument('--compression', type=int, default=6, choices=range(0, 10),
                        help='PNG compression level (0-9, where 9 is maximum compression)')
    parser.add_argument('--fixed-size', action='store_true',
                        help='Use fixed 800x240 size instead of dynamic width')
    
    args = parser.parse_args()
    
    input_dir = args.input.rstrip('\\/"\'')
    output_dir = args.output.rstrip('\\/"\'')
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max files: {args.max}")
    print(f"Quality level: {args.quality}")
    print(f"Compression level: {args.compression}")
    print(f"Dynamic width: {not args.fixed_size}")
    
    df = process_crohme2019_dataset(
        input_dir, output_dir, args.max, args.quality, args.compression, 
        use_dynamic_width=not args.fixed_size
    )
    
    if df.empty:
        print("Warning: No data was processed!")
    else:
        print(f"\nSuccessfully processed {len(df)} files.")
        print(f"Images saved in {output_dir}/[train|val|test] directories")