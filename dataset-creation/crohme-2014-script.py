import os
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import glob
import random

def get_traces_data(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """Extract trace data from InkML file (handles 2 or 3 channels: X, Y, and optional T)"""
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
                if len(coords) >= 2:  # Ensure we have at least x and y (ignore timestamp if present)
                    x, y = float(coords[0]), float(coords[1])
                    trace_coords.append((x, y))
        
        traces_data.append(trace_coords)
    
    return traces_data

def extract_latex_label(inkml_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """
    Extract LaTeX label from CROHME 2014 InkML file
    
    Args:
        inkml_path: Path to InkML file
        
    Returns:
        LaTeX label string or None
    """
    try:
        tree = ET.parse(inkml_path)
        root = tree.getroot()
        
        # Find annotation with type="truth"
        for annotation in root.findall(f"{xmlns}annotation"):
            if annotation.get('type') == 'truth':
                return annotation.text
        
        return None
    except Exception as e:
        print(f"Error extracting label from {inkml_path}: {str(e)}")
        return None

def convert_to_img_dynamic(traces_data, target_height=240, min_width=200, max_width=1200, padding_percent=0.15, quality='medium'):
    """
    Convert traces to images with DYNAMIC width based on actual content
    Height is fixed, width adjusts to maintain aspect ratio
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
    
    # Calculate actual content dimensions
    content_width = max_x - min_x
    content_height = max_y - min_y
    
    # Avoid division by zero
    if content_width == 0: content_width = 1
    if content_height == 0: content_height = 1
    
    # Calculate aspect ratio
    aspect_ratio = content_width / content_height
    
    # Calculate output dimensions
    height = target_height
    width = int(target_height * aspect_ratio)
    
    # Clamp width to reasonable bounds
    width = max(min_width, min(width, max_width))
    
    # Recalculate height if we hit max_width
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

def inkml2img_robust(input_path, output_path, quality='medium', compression=6):
    """
    Convert InkML to PNG image
    
    Returns:
        Tuple of (success: bool, width: int, height: int)
    """
    try:
        traces_data = get_traces_data(input_path)
        if not traces_data:
            return False, 0, 0
        
        img, width, height = convert_to_img_dynamic(traces_data, quality=quality)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with compression
        cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        return True, width, height
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False, 0, 0

def process_crohme2014_dataset(base_dir, output_dir, max_files=None, quality='medium', compression=6):
    """
    Process CROHME 2014 dataset with train and test folders
    
    Args:
        base_dir: Base directory containing train/ and test/ folders with InkML files
        output_dir: Output directory for images and CSV
        max_files: Maximum number of files to process per split (for testing)
        quality: Quality level ('low', 'medium', 'high')
        compression: PNG compression level (0-9)
    """
    print(f"Processing CROHME 2014 dataset from: {base_dir}")
    
    if not os.path.exists(base_dir):
        print(f"ERROR: Input directory '{base_dir}' does not exist!")
        return pd.DataFrame()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Find InkML files in train and test folders
    all_files = []
    
    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split} directory not found: {split_dir}")
            continue
        
        # Find all InkML files
        inkml_files = glob.glob(os.path.join(split_dir, "*.inkml"))
        print(f"Found {len(inkml_files)} InkML files in {split}/")
        
        # Limit files if specified
        if max_files and max_files > 0:
            random.seed(42)
            random.shuffle(inkml_files)
            inkml_files = inkml_files[:max_files]
            print(f"  Limited to {len(inkml_files)} files for testing")
        
        for inkml_file in inkml_files:
            all_files.append({
                'path': inkml_file,
                'split': split
            })
    
    if not all_files:
        print("ERROR: No InkML files found!")
        return pd.DataFrame()
    
    print(f"\nProcessing {len(all_files)} InkML files...")
    print(f"Quality level: {quality}, compression: {compression}")
    
    data = []
    widths = []
    heights = []
    
    for file_info in tqdm(all_files, desc="Converting InkML to PNG"):
        try:
            inkml_path = file_info['path']
            split = file_info['split']
            
            # Extract sample ID
            sample_id = os.path.basename(inkml_path).replace('.inkml', '')
            
            # Extract LaTeX label
            latex_label = extract_latex_label(inkml_path)
            
            if not latex_label:
                print(f"Warning: No label found for {sample_id}")
                continue
            
            # Define output path
            png_filename = f"{sample_id}.png"
            png_path = os.path.join(output_dir, split, png_filename)
            
            # Convert to PNG
            success, width, height = inkml2img_robust(inkml_path, png_path, quality, compression)
            
            if not success:
                print(f"Failed to convert {sample_id}")
                continue
            
            # Track dimensions
            widths.append(width)
            heights.append(height)
            
            # Store data
            data.append({
                'filename': png_filename,
                'sample_id': sample_id,
                'label': latex_label,
                'split': split,
                'image_width': width,
                'image_height': height,
                'original_path': inkml_path
            })
            
        except Exception as e:
            print(f"Error processing {file_info['path']}: {str(e)}")
    
    if not data:
        print("No data was processed successfully!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset statistics:")
    split_stats = df['split'].value_counts()
    for split, count in split_stats.items():
        print(f"  {split}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Print dimension statistics
    if widths and heights:
        print(f"\nImage dimension statistics:")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={int(np.mean(widths))}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, avg={int(np.mean(heights))}")
    
    # Save main CSV
    csv_path = os.path.join(output_dir, "crohme2014_database.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull CSV saved to: {csv_path}")
    
    # Save split-specific CSVs
    for split_name in df['split'].unique():
        split_df = df[df['split'] == split_name]
        split_csv = os.path.join(output_dir, f"{split_name}_database.csv")
        split_df.to_csv(split_csv, index=False)
        print(f"Saved {split_name} CSV with {len(split_df)} entries: {split_csv}")
    
    # Save simplified CSV
    simplified_df = df[['filename', 'sample_id', 'label', 'split', 'image_width', 'image_height']]
    simplified_csv = os.path.join(output_dir, "crohme2014_database_simplified.csv")
    simplified_df.to_csv(simplified_csv, index=False)
    print(f"Simplified CSV saved to: {simplified_csv}")
    
    print(f"\nDatabase created with {len(df)} entries")
    print(f"Images saved in: {os.path.join(output_dir, '[train|test]')}")
    print("="*60)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CROHME 2014 InkML files to PNG and create database')
    parser.add_argument('--input', type=str, required=True,
                        help='Base directory containing train/ and test/ folders')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for images and CSV')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of files to process per split (for testing)')
    parser.add_argument('--quality', type=str, default='medium', 
                        choices=['low', 'medium', 'high'],
                        help='Quality level for rendered images')
    parser.add_argument('--compression', type=int, default=6, choices=range(0, 10),
                        help='PNG compression level (0-9)')
    
    args = parser.parse_args()
    
    print("CROHME 2014 Dataset Processor")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Max files per split: {args.max if args.max else 'All'}")
    print(f"Quality: {args.quality}")
    print(f"Compression: {args.compression}")
    print()
    
    df = process_crohme2014_dataset(
        args.input.rstrip('\\/"\''),
        args.output.rstrip('\\/"\''),
        args.max,
        args.quality,
        args.compression
    )
    
    if df.empty:
        print("\nERROR: No data was processed!")
    else:
        print(f"\n✓ Successfully processed {len(df)} files")