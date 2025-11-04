import os
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
import numpy as np
import cv2

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

def extract_latex_label(inkml_path, xmlns='{http://www.w3.org/2003/InkML}'):
    """
    Extract LaTeX label from CROHME 2011 InkML file
    
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

def convert_crohme2011_samples(input_dir, output_dir, max_files=10, quality='medium'):
    """
    Convert a few CROHME 2011 InkML files to images for testing
    
    Args:
        input_dir: Directory containing InkML files
        output_dir: Output directory for images
        max_files: Maximum number of files to convert
        quality: Quality level ('low', 'medium', 'high')
    """
    print(f"Looking for InkML files in: {input_dir}")
    
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' does not exist!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find InkML files
    import glob
    inkml_files = glob.glob(os.path.join(input_dir, "*.inkml"))
    
    if not inkml_files:
        print(f"No InkML files found in {input_dir}")
        return
    
    # Limit number of files
    inkml_files = inkml_files[:max_files]
    
    print(f"\nConverting {len(inkml_files)} InkML files to images...")
    print(f"Quality level: {quality}")
    
    data = []
    
    for inkml_file in tqdm(inkml_files, desc="Converting"):
        try:
            # Extract sample ID
            sample_id = os.path.basename(inkml_file).replace('.inkml', '')
            
            # Extract LaTeX label
            latex_label = extract_latex_label(inkml_file)
            
            # Get traces
            traces_data = get_traces_data(inkml_file)
            
            if not traces_data:
                print(f"No traces found in {inkml_file}")
                continue
            
            # Convert to image
            img, width, height = convert_to_img_dynamic(traces_data, quality=quality)
            
            # Save image
            output_path = os.path.join(output_dir, f"{sample_id}.png")
            cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            
            # Store data
            data.append({
                'filename': f"{sample_id}.png",
                'sample_id': sample_id,
                'label': latex_label,
                'width': width,
                'height': height,
                'original_file': inkml_file
            })
            
            print(f"\n✓ Converted: {sample_id}")
            print(f"  Label: {latex_label}")
            print(f"  Dimensions: {width}x{height}")
            
        except Exception as e:
            print(f"Error processing {inkml_file}: {str(e)}")
    
    # Create CSV
    if data:
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, "converted_samples.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Successfully converted {len(data)} files")
        print(f"Images saved to: {output_dir}")
        print(f"CSV saved to: {csv_path}")
        print(f"\nDimension statistics:")
        print(f"  Width:  min={df['width'].min()}, max={df['width'].max()}, avg={int(df['width'].mean())}")
        print(f"  Height: min={df['height'].min()}, max={df['height'].max()}, avg={int(df['height'].mean())}")
        print(f"{'='*60}")
    else:
        print("No files were converted successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CROHME 2011 InkML to PNG converter')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing InkML files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for images')
    parser.add_argument('--max', type=int, default=10,
                        help='Maximum number of files to convert (default: 10)')
    parser.add_argument('--quality', type=str, default='medium', 
                        choices=['low', 'medium', 'high'],
                        help='Quality level for rendered images')
    
    args = parser.parse_args()
    
    convert_crohme2011_samples(
        args.input.rstrip('\\/"\''),
        args.output.rstrip('\\/"\''),
        args.max,
        args.quality
    )