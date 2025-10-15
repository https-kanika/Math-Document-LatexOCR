import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import random
import json
from tqdm import tqdm

# Define paths to datasets
IAM_DIR = r"C:\Users\kani1\Desktop\IE643\custom-dataset\iam-handwritten-lines-binarized"
# Update this path to the correct directory where images are stored
MATHWRITING_DIR = r"C:\Users\kani1\Desktop\IE643\custom-dataset\ProcessedFullMathwriting"
#MATHWRITING_DIR=r"C:\Users\kani1\Desktop\IE643\custom-dataset\MathWritingSmall"
OUTPUT_DIR = r"C:\Users\kani1\Desktop\IE643\custom-dataset\document-dataset"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

# Load datasets
def load_datasets():
    """Load IAM and MathWriting datasets"""
    print("Loading datasets...")
    
    # Load IAM dataset
    try:
        print("Loading IAM metadata from:", os.path.join(IAM_DIR, "iam_lines_metadata.csv"))
        # Change from tab separator to comma separator
        iam_metadata = pd.read_csv(os.path.join(IAM_DIR, "iam_lines_metadata.csv"), sep=',')
        
        print(f"Loaded {len(iam_metadata)} IAM handwritten lines")
        
        # Update image_path to include the split folder
        iam_metadata['image_path'] = iam_metadata.apply(
            lambda row: os.path.join(row['split'], row['filename']), axis=1
        )
        
        # Check the unique values in the split column
        unique_splits = iam_metadata['split'].unique()
        print(f"IAM dataset split values: {unique_splits}")
        
        # Verify that image files exist
        valid_iam_entries = []
        for i, row in iam_metadata.iterrows():
            img_path = os.path.join(IAM_DIR, row['split'], row['filename'])
            if os.path.exists(img_path):
                valid_iam_entries.append(row)
            else:
                print(f"Warning: IAM image not found: {img_path}")
                
        iam_metadata = pd.DataFrame(valid_iam_entries)
        print(f"Valid IAM lines with existing image files: {len(iam_metadata)}")
        
    except Exception as e:
        print(f"Error loading IAM dataset: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for better debugging
        return pd.DataFrame(), pd.DataFrame()
    
    # Load MathWriting dataset
    try:
        math_csv_path = os.path.join(MATHWRITING_DIR, "mathwriting_database.csv")
        math_metadata = pd.read_csv(math_csv_path)
        print(f"Loaded {len(math_metadata)} math equations from CSV")
        
        # Add is_symbol column if it doesn't exist
        if 'is_symbol' not in math_metadata.columns:
            print("'is_symbol' column not found, inferring from file paths...")
            # Assume files in "symbols" folder are symbols
            math_metadata['is_symbol'] = math_metadata['filename'].apply(
                lambda x: os.path.exists(os.path.join(MATHWRITING_DIR, "symbols", x))
            )
            
        # Verify that image files exist for MathWriting dataset
        valid_math_entries = []
        for i, row in math_metadata.iterrows():
            # Determine the correct folder based on is_symbol and split
            if row.get('is_symbol') == True:
                folder = "symbols"
            else:
                folder = row.get('split', '')  # Use split value (train, test, val)
            
            # Create image path
            img_path = os.path.join(MATHWRITING_DIR, folder, row['filename'])
            
            # Check if file exists
            if os.path.exists(img_path):
                row_copy = row.copy()
                row_copy['image_path'] = os.path.join(folder, row['filename'])
                valid_math_entries.append(row_copy)
            else:
                print(f"Warning: Math image not found: {img_path}")
        
        math_metadata = pd.DataFrame(valid_math_entries)
        total_math_images = len(math_metadata)
        print(f"Found {total_math_images} valid math images")
        
        # NEW: Limit dataset to 10,000 images (or fewer if not available)
        """MAX_MATH_IMAGES = 10000
        if total_math_images > MAX_MATH_IMAGES:
            print(f"Limiting MathWriting dataset to {MAX_MATH_IMAGES} images for testing")
            
            # Stratify by symbols vs regular equations and by split
            symbols = math_metadata[math_metadata['is_symbol'] == True]
            train_eqns = math_metadata[(math_metadata['is_symbol'] == False) & (math_metadata['split'] == 'train')]
            test_eqns = math_metadata[(math_metadata['is_symbol'] == False) & (math_metadata['split'] == 'test')]
            val_eqns = math_metadata[(math_metadata['is_symbol'] == False) & (math_metadata['split'] == 'val')]
            
            # Calculate sample sizes to maintain proportion (with minimum counts)
            symbol_count = min(len(symbols), int(MAX_MATH_IMAGES * 0.2))  # 20% symbols
            
            # Distribute remaining images proportionally among splits
            remaining = MAX_MATH_IMAGES - symbol_count
            total_eqns = len(train_eqns) + len(test_eqns) + len(val_eqns)
            
            if total_eqns > 0:
                train_prop = len(train_eqns) / total_eqns
                test_prop = len(test_eqns) / total_eqns
                val_prop = len(val_eqns) / total_eqns
            else:
                train_prop, test_prop, val_prop = 0.6, 0.3, 0.1  # Default proportions
            
            train_count = min(len(train_eqns), int(remaining * train_prop))
            test_count = min(len(test_eqns), int(remaining * test_prop))
            val_count = min(len(val_eqns), int(remaining * val_prop))
            
            # Sample from each group
            sampled_symbols = symbols.sample(symbol_count) if symbol_count > 0 else pd.DataFrame()
            sampled_train = train_eqns.sample(train_count) if train_count > 0 else pd.DataFrame()
            sampled_test = test_eqns.sample(test_count) if test_count > 0 else pd.DataFrame()
            sampled_val = val_eqns.sample(val_count) if val_count > 0 else pd.DataFrame()
            
            # Combine all samples
            math_metadata = pd.concat([sampled_symbols, sampled_train, sampled_test, sampled_val])
            
            print(f"Sampled dataset composition:")
            print(f"  - Symbols: {len(sampled_symbols)}")
            print(f"  - Train equations: {len(sampled_train)}")
            print(f"  - Test equations: {len(sampled_test)}")
            print(f"  - Val equations: {len(sampled_val)}")
            print(f"  - Total: {len(math_metadata)}")"""
        
        # For document creation, use all available data (but now limited to 10,000)
        train_math_metadata = math_metadata
        
        # Count symbols for reporting
        symbol_count = sum(1 for _, row in train_math_metadata.iterrows() 
                           if row.get('is_symbol') == True)
        
        print(f"Final dataset has {len(train_math_metadata)} images with {symbol_count} symbols")    
    except Exception as e:
        print(f"Error loading math dataset: {e}")
        print(f"Details: {str(e)}")
        return iam_metadata, pd.DataFrame()
    
    # Display first few rows of each dataset for verification
    print("\nSample IAM metadata:")
    print(iam_metadata[['filename', 'text', 'split', 'image_path']].head())
    
    print("\nSample Math metadata:")
    print(train_math_metadata[['filename', 'label', 'normalized_label', 'is_symbol', 'split', 'image_path']].head())
    
    return iam_metadata, train_math_metadata



def insert_math_in_text(iam_metadata, math_metadata, probability=0.15):
    """
    Create a function that randomly splits text sentences and inserts math equations
    
    Args:
        iam_metadata: DataFrame containing IAM text line data
        math_metadata: DataFrame containing math equation data
        probability: Chance (0-1) that a text line will have a math equation inserted
        
    Returns:
        List of combined text-math elements to be placed in the document
    """
    # Filter out non-symbol math equations
    regular_math_entries = math_metadata[math_metadata.get('is_symbol', False) == False]
    
    # If no math equations available, return original text
    if len(regular_math_entries) == 0:
        return []
    
    # Create list for combined elements
    combined_elements = []
    
    # Process each text line
    for _, text_row in iam_metadata.iterrows():
        text = text_row['text']
        
        # Only consider text lines with enough words to split
        words = text.split()
        
        if len(words) >= 6 and random.random() < probability:
            # Randomly choose a split point (avoid splitting at beginning or end)
            split_point = random.randint(2, len(words) - 2)
            
            # Create the two parts of text
            first_part = ' '.join(words[:split_point])
            second_part = ' '.join(words[split_point:])
            
            # Select a random math equation
            math_idx = random.randint(0, len(regular_math_entries) - 1)
            math_row = regular_math_entries.iloc[math_idx]
            
            # Create three elements: first text, math, second text
            combined_elements.append(("text_part", text_row, first_part))
            combined_elements.append(("math", math_row))
            combined_elements.append(("text_part", text_row, second_part))
        else:
            # Keep the original text line
            combined_elements.append(("text", text_row))
    
    return combined_elements

def insert_symbol_in_text(text, symbol_info):
        """Insert a symbol within text at a random position between words"""
        words = text.split()
        if len(words) <= 3:  # Too short, don't modify
            return None, None
        
        # Choose a position between words (not at start or end)
        insert_pos = random.randint(1, len(words) - 2)
        
        # Split the text
        first_part = ' '.join(words[:insert_pos])
        second_part = ' '.join(words[insert_pos:])
        
        return first_part, second_part, symbol_info
    

def create_document(iam_metadata, math_metadata, doc_id, num_elements=20, visualize=False):
    """
    Create an A4 document with handwritten text and math equations with varied alignment,
    occasionally adding math symbols after text lines and inserting math equations mid-sentence.
    """
    # First, check if we have data to work with
    if len(iam_metadata) == 0:
        raise ValueError("No IAM text lines available. Cannot create document.")
    
    # A4 size in pixels at 300 DPI: 2480 x 3508 pixels (portrait)
    a4_width, a4_height = 2480,3580
    
    # Create blank white document
    doc_image = np.ones((a4_height, a4_width, 3), dtype=np.uint8) * 255  # RGB (3 channels)
    
    # Create blank annotation image as a copy of the document image
    annotation_image = doc_image.copy()
    
    # Initialize document labels
    doc_labels = {
        "document_id": doc_id,
        "width": a4_width,
        "height": a4_height,
        "elements": [],
        "latex_document": "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n\n"
    }
    
    # Define smaller margins to fill more of the page
    margin_top, margin_bottom = 80, 80
    margin_left, margin_right = 120, 120
    line_spacing = 15
    
    # Current y position
    y_position = margin_top
    
    # Initialize symbol paths list
    symbol_paths = []
    
    # Determine if we have math equations available
    has_math_equations = False
    regular_math_entries = pd.DataFrame()
    
    # Check for symbols in the dataset
    if 'is_symbol' in math_metadata.columns:
        # Get symbol images from math_metadata where is_symbol=True
        symbol_entries = math_metadata[math_metadata['is_symbol'] == True]
        
        # If we have symbol entries in metadata, use those
        if len(symbol_entries) > 0:
            for _, row in symbol_entries.iterrows():
                symbol_path = os.path.join(MATHWRITING_DIR, "symbols", row['filename'])
                if os.path.exists(symbol_path):
                    # FIX: Choose the best available name for the symbol
                    if pd.notna(row.get('normalized_label')) and row.get('normalized_label'):
                        symbol_name = row.get('normalized_label')
                    elif pd.notna(row.get('label')) and row.get('label'):
                        symbol_name = row.get('label')
                    else:
                        symbol_name = f"sym_{row['filename'].split('.')[0]}"
                        
                    symbol_paths.append({
                        'path': symbol_path,
                        'name': symbol_name
                    })
        
        # Get regular math equations (non-symbols)
        regular_math_entries = math_metadata[math_metadata['is_symbol'] == False]
        has_math_equations = len(regular_math_entries) > 0
    else:
        # If no is_symbol column, consider all math entries as regular equations
        regular_math_entries = math_metadata
        has_math_equations = len(regular_math_entries) > 0
    
    # If we still don't have symbols, look directly in the symbols folder
    if not symbol_paths:
        symbols_dir = os.path.join(MATHWRITING_DIR, "symbols")
        if os.path.exists(symbols_dir):
            for f in os.listdir(symbols_dir):
                if f.endswith('.png') or f.endswith('.jpg'):
                    symbol_path = os.path.join(symbols_dir, f)
                    # FIX: Make sure we have a valid name
                    symbol_name = f"sym_{f.split('.')[0]}"
                    
                    symbol_paths.append({
                        'path': symbol_path, 
                        'name': symbol_name
                    })
    
    print(f"Found {len(symbol_paths)} math symbols to use")
    if not has_math_equations:
        print("WARNING: No regular math equations found. Document will only contain text.")
    
    # Decide if this document will use mid-sentence math equations (30% chance)
    use_mid_sentence_math = random.random() < 0.3 and has_math_equations
    
    # Create list of elements to place
    elements = []
    max_elements = max(num_elements, 60)  # Increased to ensure we have enough elements to fill the page
    
    # Track the current text line height for mid-sentence math sizing
    current_text_height = None
    
    if use_mid_sentence_math:
        print(f"Creating document {doc_id} with mid-sentence math equations")
        
        # Select random subset of IAM text lines to work with
        sample_size = min(max_elements // 2, len(iam_metadata))  # Use half the max elements as text lines
        text_sample_indices = random.sample(range(len(iam_metadata)), sample_size)
        text_samples = iam_metadata.iloc[text_sample_indices]
        
        # Generate combined elements with math inserted into text
        combined_elements = insert_math_in_text(text_samples, math_metadata, probability=0.4)
        
        # Process elements to store context for sizing
        processed_elements = []
        i = 0
        while i < len(combined_elements):
            element = combined_elements[i]
            
            if element[0] == "text_part" and i+2 < len(combined_elements):
                # Check if this is the first part of a text-math-text sequence
                if (combined_elements[i+1][0] == "math" and 
                    combined_elements[i+2][0] == "text_part"):
                    
                    # Store them together as a group for proper sizing
                    processed_elements.append({
                        "type": "text_math_text",
                        "first_part": element,
                        "math": combined_elements[i+1],
                        "second_part": combined_elements[i+2]
                    })
                    i += 3  # Skip the next two elements since we processed them
                else:
                    processed_elements.append({"type": "single", "element": element})
                    i += 1
            else:
                processed_elements.append({"type": "single", "element": element})
                i += 1
        
        # Now expand processed elements back
        for item in processed_elements:
            if item["type"] == "single":
                elements.append(item["element"])
            else:
                # Add as individual elements but with grouping info
                first_part = item["first_part"]
                math_part = item["math"]
                second_part = item["second_part"]
                
                # Add a special tuple format that includes grouping information
                elements.append(("text_part", first_part[1], first_part[2], "group_start"))
                elements.append(("math", math_part[1], "mid_sentence"))
                elements.append(("text_part", second_part[1], second_part[2], "group_end"))
        
        # If we need more elements, add regular ones
        if len(elements) < max_elements:
            additional_needed = max_elements - len(elements)
            for i in range(additional_needed):
                # 60% text, 40% math
                is_text = random.random() < 0.6
                if is_text:
                    text_idx = random.randint(0, len(iam_metadata) - 1)
                    text_row = iam_metadata.iloc[text_idx]
                    elements.append(("text", text_row))
                elif has_math_equations:
                    math_idx = random.randint(0, len(regular_math_entries) - 1)
                    math_row = regular_math_entries.iloc[math_idx]
                    elements.append(("math", math_row))
    else:
        # Fill elements list with random text and math entries (original approach)
        for i in range(max_elements):
            is_text = True if not has_math_equations else (random.random() < 0.6)
            if is_text:
                # Select random text line
                text_idx = random.randint(0, len(iam_metadata) - 1)
                text_row = iam_metadata.iloc[text_idx]
                elements.append(("text", text_row))
            else:
                # We've already confirmed we have math equations available
                math_idx = random.randint(0, len(regular_math_entries) - 1)
                math_row = regular_math_entries.iloc[math_idx]
                elements.append(("math", math_row))
    
    # Place elements on document
    placed_elements = 0
    
    # Keep track of text height for proper math sizing in mid-sentence
    current_text_height = None
    current_group_elements = []
    in_text_math_group = False
    
    for i, element_data in enumerate(elements):
        try:
            element_type = element_data[0]
            
            # Handle text part (from a split text line)
            if element_type == "text_part":
                # Check if this is part of a group (for mid-sentence math)
                is_group_start = False
                is_group_end = False
                
                if len(element_data) > 3:  # Has group info
                    if element_data[3] == "group_start":
                        is_group_start = True
                        in_text_math_group = True
                        current_group_elements = []
                    elif element_data[3] == "group_end":
                        is_group_end = True
                
                _, elem_row, text_part = element_data[:3]
                
                # Load IAM text image
                img_path = os.path.join(IAM_DIR, elem_row['image_path'])
                full_img = cv2.imread(img_path)
                
                if full_img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                
                # Convert BGR to RGB
                full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

                # If this is the start of a group, save the text height for math sizing
                if is_group_start:
                    current_text_height = full_img.shape[0]
                
                # Get the partial text
                label = text_part
                
                # Calculate approximate width based on character ratio
                full_text = elem_row['text']
                
                # Determine if this is first or second part
                words = full_text.split()
                first_part_words = text_part.split()
                is_first_part = ' '.join(words[:len(first_part_words)]) == text_part
                
                # Approximate width ratio based on character count
                width_ratio = max(0.2, min(0.8, len(text_part) / max(1, len(full_text))))
                
                # Create a simplified text segment image (a crude approximation)
                if is_first_part:
                    # First part - take left portion
                    new_width = max(int(full_img.shape[1] * width_ratio), 50)
                    elem_img = full_img[:, :new_width].copy()
                else:
                    # Second part - take right portion
                    new_width = max(int(full_img.shape[1] * width_ratio), 50)
                    start_x = max(0, full_img.shape[1] - new_width)
                    elem_img = full_img[:, start_x:].copy()
                
                # Escape special LaTeX characters in text
                latex_label = escape_latex_text(label)
                
                # Set element type for visualization and labeling
                element_type = "text"
                
                # If we're in a group, collect this element
                if in_text_math_group:
                    current_group_elements.append({
                        "type": "text_part",
                        "image": elem_img,
                        "label": label,
                        "latex_label": latex_label
                    })
                    
                    # If this is the end of the group, we'll handle placement later
                    if is_group_end:
                        in_text_math_group = False
                        continue
                
            # Handle regular text
            elif element_type == "text":
                elem_row = element_data[1]
                
                # Load IAM text image
                img_path = os.path.join(IAM_DIR, elem_row['image_path'])
                elem_img = cv2.imread(img_path)
                
                if elem_img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                    
                # Convert BGR to RGB
                elem_img = cv2.cvtColor(elem_img, cv2.COLOR_BGR2RGB)
               
                # Update current text height for potential future mid-sentence math
                current_text_height = elem_img.shape[0]
                
                # Get label
                label = elem_row['text']
                
                # Escape special LaTeX characters in text
                latex_label = escape_latex_text(label)
                
                # Set element type
                element_type = "text"
                
                # 20% chance to insert a symbol in the middle of text
                add_inline_symbol = random.random() < 0.2 and symbol_paths and len(label.split()) > 3
                symbol_data = None
                
                if add_inline_symbol and symbol_paths:
                    # Select a random symbol
                    symbol_info = random.choice(symbol_paths)
                    symbol_path = symbol_info['path']
                    
                    # FIX: Ensure we have a valid symbol name
                    if 'name' in symbol_info and symbol_info['name'] and not pd.isna(symbol_info['name']):
                        symbol_name = symbol_info['name']
                    else:
                        # Extract name from file path as fallback
                        symbol_name = f"sym_{os.path.basename(symbol_path).split('.')[0]}"
                    
                    try:
                        # Split the text
                        first_part, second_part, _ = insert_symbol_in_text(label, symbol_info)
                        
                        # Load the text image
                        
                        elem_img = cv2.cvtColor(elem_img, cv2.COLOR_BGR2RGB)
                                                
                        # Load symbol image
                        symbol_img = cv2.imread(symbol_path)
                        symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2RGB)
                        
                        # Calculate appropriate height for the symbol
                        text_height = elem_img.shape[0]
                        scale_factor = text_height / symbol_img.shape[0]
                        new_height = text_height
                        new_width = int(symbol_img.shape[1] * scale_factor)
                        symbol_img = cv2.resize(symbol_img, (new_width, new_height))
                        
                        # Split the text image proportionally to text length
                        text_width = elem_img.shape[1]
                        split_ratio = len(first_part) / len(label)
                        split_point = int(text_width * split_ratio)
                        
                        # Create first part image
                        first_img = elem_img[:, :split_point].copy()
                        
                        # Create second part image
                        second_img = elem_img[:, split_point:].copy()
                        
                        # Create a combined image
                        combined_width = first_img.shape[1] + symbol_img.shape[1] + second_img.shape[1] + 10  # 5px spacing on each side
                        combined_height = max(first_img.shape[0], symbol_img.shape[0], second_img.shape[0])
                        combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
                        
                        # Place images
                        x_pos = 0
                        combined_img[0:first_img.shape[0], x_pos:x_pos+first_img.shape[1]] = first_img
                        
                        x_pos += first_img.shape[1] + 5  # Add spacing
                        combined_img[0:symbol_img.shape[0], x_pos:x_pos+symbol_img.shape[1]] = symbol_img
                        
                        x_pos += symbol_img.shape[1] + 5  # Add spacing
                        combined_img[0:second_img.shape[0], x_pos:x_pos+second_img.shape[1]] = second_img
                        
                        # Replace the original image
                        elem_img = combined_img
                        
                        # Update element type and data
                        element_type = "text_with_inline_symbol"
                        
                        # Create a combined label
                        latex_label = f"{escape_latex_text(first_part)} ${symbol_name}$ {escape_latex_text(second_part)}"
                        label = f"{first_part} {symbol_name} {second_part}"
                        
                        # Store symbol info for annotation
                        symbol_data = {
                            "path": symbol_path,
                            "name": symbol_name,
                            "width": new_width,
                            "height": new_height,
                            "first_text": first_part,
                            "second_text": second_part,
                            "first_width": first_img.shape[1],
                            "second_width": second_img.shape[1]
                        }
                        
                    except Exception as e:
                        print(f"Warning: Failed to add inline symbol: {str(e)}")
                        # Fall back to normal text handling
                        add_inline_symbol = False
                
                # If not adding an inline symbol, check if we should add an end-of-line symbol
                if not add_inline_symbol:
                    add_symbol = random.random() < 0.2 and symbol_paths and label.strip().endswith('.')
                    if add_symbol and symbol_paths:
                        # Select a random symbol
                        symbol_info = random.choice(symbol_paths)
                        symbol_path = symbol_info['path']
                        
                        # FIX: Ensure we have a valid symbol name
                        if 'name' in symbol_info and symbol_info['name'] and not pd.isna(symbol_info['name']):
                            symbol_name = symbol_info['name']
                        else:
                            # Extract name from file path as fallback
                            symbol_name = f"sym_{os.path.basename(symbol_path).split('.')[0]}"
                   
                        try:
                            # Load symbol image
                            symbol_img = cv2.imread(symbol_path)
                            symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2RGB)
                            
                            # Calculate height ratio to match text line height
                            text_height = elem_img.shape[0]
                            symbol_height = symbol_img.shape[0]
                            scale_factor = text_height / symbol_height
                            
                            # Resize symbol to match text height
                            new_height = text_height
                            new_width = int(symbol_img.shape[1] * scale_factor)
                            symbol_img = cv2.resize(symbol_img, (new_width, new_height))
                            
                            # Create a new composite image with text and symbol
                            composite_width = elem_img.shape[1] + symbol_img.shape[1] + 10  # 10px spacing
                            composite_height = max(elem_img.shape[0], symbol_img.shape[0])
                            composite_img = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 255
                            
                            # Place text and symbol in composite image
                            composite_img[0:elem_img.shape[0], 0:elem_img.shape[1]] = elem_img
                            composite_img[0:symbol_img.shape[0], 
                                        elem_img.shape[1]+10:elem_img.shape[1]+10+symbol_img.shape[1]] = symbol_img
                            
                            # Replace the text image with the composite image
                            elem_img = composite_img
                            
                            # Store symbol data for separate labeling
                            symbol_data = {
                                "path": symbol_path,
                                "name": symbol_name,
                                "width": new_width,
                                "height": new_height
                            }
                            
                            # Update element type to indicate symbol was added
                            element_type = "text_with_symbol"
                            
                        except Exception as e:
                            print(f"Warning: Failed to add symbol from {symbol_path}: {str(e)}")
                            # Fall back to normal text handling

            # Handle math equation
            elif element_type == "math":
                # Check if this is mid-sentence math
                is_mid_sentence = False
                if len(element_data) > 2 and element_data[2] == "mid_sentence":
                    is_mid_sentence = True
                
                elem_row = element_data[1]
                
                # Load math equation image - only from train folder
                img_path = os.path.join(MATHWRITING_DIR, elem_row['image_path'])
                
                # Try loading the image
                elem_img = cv2.imread(img_path)
                
                # If loading failed, try only train directory
                if elem_img is None:
                    print(f"Warning: Could not load image {img_path}")
                    
                    # Only try train directory
                    alt_path = os.path.join(MATHWRITING_DIR, "train", os.path.basename(elem_row['image_path']))
                    if os.path.exists(alt_path):
                        elem_img = cv2.imread(alt_path)
                        if elem_img is not None:
                            img_path = alt_path  # Update path if successful
                
                # If still no image, skip this element
                if elem_img is None:
                    print(f"Warning: Could not load any version of the image")
                    continue
                
                # Convert BGR to RGB

                elem_img = cv2.cvtColor(elem_img, cv2.COLOR_BGR2RGB)

                # Get label (normalized_label if available, otherwise label)
                # FIX: Ensure we have a valid math label
                if pd.notna(elem_row.get('normalized_label')) and elem_row.get('normalized_label'):
                    label = elem_row.get('normalized_label')
                elif pd.notna(elem_row.get('label')) and elem_row.get('label'):
                    label = elem_row.get('label')
                else:
                    # Use a generic label if nothing else available
                    label = f"eq_{os.path.basename(img_path).split('.')[0]}"
                
                # For math, use label as is (assuming it's already LaTeX)
                latex_label = label
                
                # Set element type
                element_type = "equation"
                
                # Different scaling for mid-sentence vs standalone math
                if is_mid_sentence and current_text_height is not None:
                    # Scale to match the text height
                    scale_factor = current_text_height / elem_img.shape[0]
                    new_height = current_text_height
                    new_width = int(elem_img.shape[1] * scale_factor)
                    elem_img = cv2.resize(elem_img, (new_width, new_height))
                    
                    # Add to current group
                    current_group_elements.append({
                        "type": "math",
                        "image": elem_img,
                        "label": label,
                        "latex_label": latex_label
                    })
                    continue  # Skip individual placement for group elements
                else:
                    # Apply standard scaling to standalone math equations
                    math_scale_factor = 0.35  # Further reduced to fit more on the page
                    elem_height, elem_width = elem_img.shape[0], elem_img.shape[1]
                    new_width = int(elem_width * math_scale_factor)
                    new_height = int(elem_height * math_scale_factor)
                    elem_img = cv2.resize(elem_img, (new_width, new_height))
            
            # Handle placement of text-math-text group
            if not in_text_math_group and current_group_elements and len(current_group_elements) == 3:
                # We have a complete text-math-text group to place
                first_text = current_group_elements[0]
                math_eq = current_group_elements[1]
                second_text = current_group_elements[2]
                
                # FIX: Ensure math equation has valid label
                if pd.isna(math_eq["label"]) or math_eq["label"] == "nan" or not math_eq["label"]:
                    math_eq["label"] = f"eq_{hash(str(math_eq)) % 1000}"
                    math_eq["latex_label"] = math_eq["label"]
                
                # Create a combined image with all three elements
                combined_height = max(first_text["image"].shape[0], 
                                      math_eq["image"].shape[0], 
                                      second_text["image"].shape[0])
                
                # Calculate total width
                spacing = 5  # Pixels between elements
                combined_width = (first_text["image"].shape[1] + spacing + 
                                 math_eq["image"].shape[1] + spacing + 
                                 second_text["image"].shape[1])
                
                # Create blank combined image
                # Create blank combined image
                combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
                
                # Place first text
                x_pos = 0
                h1 = first_text["image"].shape[0]
                w1 = first_text["image"].shape[1]
                combined_img[0:h1, x_pos:x_pos+w1] = first_text["image"]
                
                # Place math
                x_pos += w1 + spacing
                h2 = math_eq["image"].shape[0]
                w2 = math_eq["image"].shape[1]
                combined_img[0:h2, x_pos:x_pos+w2] = math_eq["image"]
                
                # Place second text
                x_pos += w2 + spacing
                h3 = second_text["image"].shape[0]
                w3 = second_text["image"].shape[1]
                combined_img[0:h3, x_pos:x_pos+w3] = second_text["image"]
                
                # Use the combined image as our element
                elem_img = combined_img
                element_type = "text_with_math"
                
                # Create combined label and LaTeX with valid math
                label = f"{first_text['label']} {math_eq['label']} {second_text['label']}"
                latex_label = f"{first_text['latex_label']} ${math_eq['latex_label']}$ {second_text['latex_label']}"
                
                # Clear the group
                current_group_elements = []
            
            # If we're still collecting group elements, skip placement
            if in_text_math_group:
                continue
                
            # Scale image to fit width if needed (applies to both text and math)
            max_width = a4_width - margin_left - margin_right
            elem_height, elem_width = elem_img.shape[0], elem_img.shape[1]
            
            if elem_width > max_width:
                scale_factor = max_width / elem_width
                new_width = int(elem_width * scale_factor)
                new_height = int(elem_height * scale_factor)
                elem_img = cv2.resize(elem_img, (new_width, new_height))
                elem_height, elem_width = new_height, new_width
            
            # Check if element fits on current page
            if y_position + elem_height > a4_height - margin_bottom:
                # Not enough vertical space left, we've filled the page
                break
            
            # Randomly determine alignment style
            alignment_type = random.choice(["left", "center", "random_shift"])
            
            # Calculate x position based on alignment
            if alignment_type == "left":
                # Left alignment (default)
                x_position = margin_left
                alignment_latex = "\\noindent "  # No indentation for left alignment
            elif alignment_type == "center":
                # Center alignment
                x_position = (a4_width - elem_width) // 2
                alignment_latex = "\\begin{center}\n"  # Center environment in LaTeX
            else:  # random_shift
                # Random shift from left margin
                max_shift = min(300, max_width - elem_width)  # Limit shift to keep within page
                x_position = margin_left + random.randint(0, max(0, max_shift))
                # Use hspace for random shifts
                shift_amount = (x_position - margin_left) / 10  # Convert pixels to approx ems
                alignment_latex = f"\\noindent\\hspace{{{shift_amount}em}}"
            
            # Calculate bounding box
            x1, y1 = x_position, y_position
            x2, y2 = x_position + elem_width, y_position + elem_height
            
            # Place element on document
            doc_image[y1:y2, x1:x2] = elem_img
            
            # Place element on annotation image as well
            annotation_image[y1:y2, x1:x2] = elem_img
            
            # Draw bounding box on annotation image
            if element_type == "text_with_symbol":
                # Draw separate boxes for text and symbol
                text_width = elem_img.shape[1] - symbol_data["width"] - 10  # Subtract symbol width and gap
                
                # Text bounding box (green)
                text_color = (0, 255, 0)  # Green for text
                cv2.rectangle(annotation_image, (x1, y1), (x1 + text_width, y2), text_color, 2)
                cv2.putText(annotation_image, "text", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                # Symbol bounding box (blue)
                symbol_color = (255, 0, 0)  # Blue for symbol
                symbol_x1 = x1 + text_width + 10  # Add gap
                cv2.rectangle(annotation_image, (symbol_x1, y1), (x2, y2), symbol_color, 2)
                cv2.putText(annotation_image, "symbol", (symbol_x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, symbol_color, 1)
            elif element_type == "text_with_inline_symbol":
                # Calculate the scaled widths for each part
                spacing = 5
                first_width = symbol_data["first_width"]
                symbol_width = symbol_data["width"]
                second_width = symbol_data["second_width"]
                
                # Calculate total original width
                total_orig_width = first_width + spacing + symbol_width + spacing + second_width
                
                # Calculate scaling if it was applied
                if total_orig_width > max_width:
                    scale_factor = elem_width / total_orig_width
                    first_width_scaled = int(first_width * scale_factor)
                    symbol_width_scaled = int(symbol_width * scale_factor)
                    second_width_scaled = elem_width - first_width_scaled - symbol_width_scaled - (2 * spacing)
                else:
                    first_width_scaled = first_width
                    symbol_width_scaled = symbol_width
                    second_width_scaled = second_width
                
                # Calculate positions for annotation
                symbol_x1 = x1 + first_width_scaled + spacing
                symbol_x2 = symbol_x1 + symbol_width_scaled
                second_x1 = symbol_x2 + spacing
                
                # Draw bounding boxes with the corrected coordinates
                text_color = (0, 255, 0)  # Green for text
                symbol_color = (255, 0, 0)  # Blue for symbol
                
                # First text part
                cv2.rectangle(annotation_image, (x1, y1), (x1 + first_width_scaled, y2), text_color, 2)
                cv2.putText(annotation_image, "text", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                # Symbol
                cv2.rectangle(annotation_image, (symbol_x1, y1), (symbol_x2, y2), symbol_color, 2)
                cv2.putText(annotation_image, "symbol", (symbol_x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, symbol_color, 1)
                
                # Second text part
                cv2.rectangle(annotation_image, (second_x1, y1), (x2, y2), text_color, 2)
                cv2.putText(annotation_image, "text", (second_x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            elif element_type == "text_with_math":
                # Store original dimensions before any scaling/placement
                first_width_orig = first_text["image"].shape[1]
                math_width_orig = math_eq["image"].shape[1]
                second_width_orig = second_text["image"].shape[1]
                spacing = 5
                
                # Calculate the actual total width before any scaling
                total_width_orig = first_width_orig + spacing + math_width_orig + spacing + second_width_orig
                
                # Calculate the actual scaling that was applied to fit the page width
                if total_width_orig > max_width:
                    # If image was scaled down to fit page
                    scale_factor = elem_width / total_width_orig
                    # Apply this scale factor to individual element widths
                    first_width_scaled = int(first_width_orig * scale_factor)
                    math_width_scaled = int(math_width_orig * scale_factor)
                    second_width_scaled = elem_width - first_width_scaled - math_width_scaled - (2 * spacing)
                    
                    # Calculate exact positions for annotation
                    math_x1 = x1 + first_width_scaled + spacing
                    math_x2 = math_x1 + math_width_scaled
                    second_x1 = math_x2 + spacing
                else:
                    # No scaling was needed
                    first_width_scaled = first_width_orig
                    math_width_scaled = math_width_orig
                    
                    # Calculate exact positions for annotation
                    math_x1 = x1 + first_width_scaled + spacing
                    math_x2 = math_x1 + math_width_scaled
                    second_x1 = math_x2 + spacing
                
                # Draw bounding boxes with the corrected coordinates
                text_color = (0, 255, 0)  # Green for text
                math_color = (0, 0, 255)  # Red for math
                
                # First text part
                cv2.rectangle(annotation_image, (x1, y1), (x1 + first_width_scaled, y2), text_color, 2)
                cv2.putText(annotation_image, "text", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                # Math part
                cv2.rectangle(annotation_image, (math_x1, y1), (math_x2, y2), math_color, 2)
                cv2.putText(annotation_image, "equation", (math_x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, math_color, 1)
                
                # Second text part
                cv2.rectangle(annotation_image, (second_x1, y1), (x2, y2), text_color, 2)
                cv2.putText(annotation_image, "text", (second_x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            else:
                color = (0, 255, 0) if element_type == "text" else (0, 0, 255)  # Green for text, Red for equations
                cv2.rectangle(annotation_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotation_image, element_type, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Create LaTeX for this element
            if element_type == "equation":
                # FIX: Final check for valid LaTeX
                if pd.isna(latex_label) or latex_label == "nan" or not latex_label:
                    latex_label = f"x_{hash(img_path) % 100}"
                
                # For equations, handle center alignment specially
                if alignment_type == "center":
                    # Math display environments are already centered, so just use them directly
                    element_latex = f"\\begin{{displaymath}}\n{latex_label}\n\\end{{displaymath}}\n\n"
                else:
                    # For left or random shift, we need to apply the alignment
                    element_latex = f"{alignment_latex}\\begin{{displaymath}}\n{latex_label}\n\\end{{displaymath}}\n\n"
            # Modify the LaTeX generation for text_with_math elements
            elif element_type == "text_with_math":
                # FIX: Final check for valid LaTeX in the math part
                if pd.isna(math_eq["latex_label"]) or math_eq["latex_label"] == "nan" or not math_eq["latex_label"]:
                    math_eq["latex_label"] = f"x_{hash(str(math_eq)) % 100}"
                
                # For text with embedded math, create inline math
                # Use inline math with $...$ instead of $$...$$ to avoid line breaks
                latex_label = f"{first_text['latex_label']} ${math_eq['latex_label']}$ {second_text['latex_label']}"
                
                if alignment_type == "center":
                    element_latex = f"\\begin{{center}}\n{latex_label}\n\\end{{center}}\n\n"
                else:
                    element_latex = f"{alignment_latex}{latex_label}\n\n"
            # Handle text with symbols and inline symbols elements
            elif element_type == "text_with_symbol" or element_type == "text_with_inline_symbol":
                # FIX: Final check for valid symbol name
                if pd.isna(latex_label) or "nan" in latex_label:
                    if element_type == "text_with_symbol":
                        # Fix the symbol name in the data
                        symbol_data["name"] = f"sym_{hash(symbol_data['path']) % 1000}"
                        # Recreate the latex_label
                        latex_label = escape_latex_text(label) + " $" + symbol_data["name"] + "$"
                    else:  # text_with_inline_symbol
                        # Extract the parts around "nan"
                        parts = latex_label.split(" $nan$ ")
                        if len(parts) == 2:
                            symbol_name = f"sym_{hash(symbol_data['path']) % 1000}"
                            symbol_data["name"] = symbol_name
                            latex_label = f"{parts[0]} ${symbol_name}$ {parts[1]}"
                
                if alignment_type == "center":
                    element_latex = f"\\begin{{center}}\n{latex_label}\n\\end{{center}}\n\n"
                else:
                    element_latex = f"{alignment_latex}{latex_label}\n\n"
                
            else:  # regular text
                # For text, just use the text with appropriate alignment
                if alignment_type == "center":
                    element_latex = f"\\begin{{center}}\n{latex_label}\n\\end{{center}}\n\n"
                else:
                    element_latex = f"{alignment_latex}{latex_label}\n\n"
            
            # Add to document LaTeX
            doc_labels["latex_document"] += element_latex
            
            # Add element to labels
            element_data = {
                "id": placed_elements,
                "type": element_type,
                "text": label,
                "latex": latex_label,
                "bbox": [x1, y1, x2, y2],
                "alignment": alignment_type,
                "original_source": os.path.basename(img_path)
            }

            # If this has a symbol at the end of text, add symbol information
            if element_type == "text_with_symbol" and symbol_data:
                # Calculate text and symbol bounding boxes
                text_width = elem_img.shape[1] - symbol_data["width"] - 10  # Subtract symbol width and gap
                
                # Update the main element to be text only
                element_data["type"] = "text"  # Change from text_with_symbol to just text
                element_data["bbox"] = [x1, y1, x1 + text_width, y2]  # Update bbox to only cover text portion
                element_data["has_associated_symbol"] = True
                element_data["associated_symbol_id"] = f"{placed_elements}_symbol"
                
                # Calculate symbol bounding box
                symbol_x1 = x1 + text_width + 10  # Text width plus gap
                symbol_y1 = y1
                symbol_x2 = x2
                symbol_y2 = y2
                
                # FIX: Ensure symbol name is not "nan"
                if pd.isna(symbol_data["name"]) or symbol_data["name"] == "nan":
                    symbol_data["name"] = f"sym_{hash(symbol_data['path']) % 1000}"
                
                # Add symbol as a separate element
                doc_labels["elements"].append({
                    "id": f"{placed_elements}_symbol",
                    "type": "symbol",
                    "text": symbol_data["name"],
                    "latex": symbol_data["name"],
                    "bbox": [symbol_x1, symbol_y1, symbol_x2, symbol_y2],
                    "alignment": alignment_type,
                    "original_source": os.path.basename(symbol_data["path"]),
                    "associated_text_id": placed_elements
                })
                
                # Add symbol info to the main element
                element_data["has_symbol"] = True
                element_data["symbol_id"] = f"{placed_elements}_symbol"
                element_data["symbol_name"] = symbol_data["name"]
            
            # If this is text with inline symbol, add annotations for the parts
            elif element_type == "text_with_inline_symbol" and symbol_data:
                # FIX: Ensure symbol name is not "nan"
                if pd.isna(symbol_data["name"]) or symbol_data["name"] == "nan":
                    symbol_data["name"] = f"sym_{hash(symbol_data['path']) % 1000}"
                
                # Store original dimensions before any scaling
                first_width_orig = symbol_data["first_width"]
                symbol_width_orig = symbol_data["width"]
                second_width_orig = symbol_data["second_width"]
                spacing = 5
                
                # Calculate the actual total width before any scaling
                total_width_orig = first_width_orig + spacing + symbol_width_orig + spacing + second_width_orig
                
                # Calculate the scaling that was applied
                if total_width_orig > max_width:
                    scale_factor = elem_width / total_width_orig
                    # Apply this scale factor to individual element widths
                    first_width_scaled = int(first_width_orig * scale_factor)
                    symbol_width_scaled = int(symbol_width_orig * scale_factor)
                    second_width_scaled = elem_width - first_width_scaled - symbol_width_scaled - (2 * spacing)
                else:
                    first_width_scaled = first_width_orig
                    symbol_width_scaled = symbol_width_orig
                    second_width_scaled = second_width_orig
                
                # Calculate exact positions for annotation
                symbol_x1 = x1 + first_width_scaled + spacing
                symbol_x2 = symbol_x1 + symbol_width_scaled
                second_x1 = symbol_x2 + spacing
                
                # Update element metadata
                element_data["has_inline_symbol"] = True
                element_data["first_text"] = symbol_data["first_text"]
                element_data["symbol"] = symbol_data["name"]
                element_data["second_text"] = symbol_data["second_text"]
                
                # Store the corrected bbox coordinates
                element_data["first_text_bbox"] = [x1, y1, x1 + first_width_scaled, y2]
                element_data["symbol_bbox"] = [symbol_x1, y1, symbol_x2, y2]
                element_data["second_text_bbox"] = [second_x1, y1, x2, y2]
            
            # If this is a text with math, add annotations for mid-sentence math
            elif element_type == "text_with_math":
                # Store original dimensions before any scaling/placement
                first_width_orig = first_text["image"].shape[1]
                math_width_orig = math_eq["image"].shape[1]
                second_width_orig = second_text["image"].shape[1]
                spacing = 5
                
                # Calculate the actual total width before any scaling
                total_width_orig = first_width_orig + spacing + math_width_orig + spacing + second_width_orig
                
                # Calculate the actual scaling that was applied to fit the page width
                if total_width_orig > max_width:
                    # If image was scaled down to fit page
                    scale_factor = elem_width / total_width_orig
                    # Apply this scale factor to individual element widths
                    first_width_scaled = int(first_width_orig * scale_factor)
                    math_width_scaled = int(math_width_orig * scale_factor)
                    second_width_scaled = elem_width - first_width_scaled - math_width_scaled - (2 * spacing)
                    
                    # Calculate exact positions for annotation
                    math_x1 = x1 + first_width_scaled + spacing
                    math_x2 = math_x1 + math_width_scaled
                    second_x1 = math_x2 + spacing
                else:
                    # No scaling was needed
                    first_width_scaled = first_width_orig
                    math_width_scaled = math_width_orig
                    
                    # Calculate exact positions for annotation
                    math_x1 = x1 + first_width_scaled + spacing
                    math_x2 = math_x1 + math_width_scaled
                    second_x1 = math_x2 + spacing
                
                # FIX: Ensure we don't have nan for math equation
                if pd.isna(math_eq["label"]) or math_eq["label"] == "nan" or not math_eq["label"]:
                    math_eq["label"] = f"eq_{hash(str(math_eq)) % 1000}"
                    
                # Update element metadata
                element_data["has_embedded_math"] = True
                element_data["first_text"] = first_text["label"]
                element_data["math"] = math_eq["label"] 
                element_data["second_text"] = second_text["label"]
                
                # Store the corrected bbox coordinates
                element_data["first_text_bbox"] = [x1, y1, x1 + first_width_scaled, y2]
                element_data["math_bbox"] = [math_x1, y1, math_x2, y2]
                element_data["second_text_bbox"] = [second_x1, y1, x2, y2]
            
            # Add main element to labels
            doc_labels["elements"].append(element_data)
            
            # Update y position for next element
            y_position = y2 + line_spacing
            placed_elements += 1
            
        except Exception as e:
            print(f"Error placing element {i}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
    
    # Close the LaTeX document
    doc_labels["latex_document"] += "\\end{document}"
    
    print(f"Placed {placed_elements} elements on document {doc_id}")
    return doc_image, annotation_image, doc_labels



def escape_latex_text(text):
    """
    Escape special characters in text for LaTeX
    """
    # Replace common LaTeX special characters
    replacements = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
        '\\': '\\textbackslash{}',
        '<': '\\textless{}',
        '>': '\\textgreater{}'
    }
    
    # Replace special characters
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def create_dataset(iam_metadata, math_metadata, num_documents=10, resize_factor=0.5, max_visualizations=5, 
                  batch_size=1000, resume_from=0):
    """
    Create a dataset of documents that fill the entire A4 page with batch saving
    
    Args:
        iam_metadata: DataFrame with IAM dataset info
        math_metadata: DataFrame with math equation data
        num_documents: Number of documents to generate
        resize_factor: Factor to resize images (e.g., 0.5 = half size)
        max_visualizations: Maximum number of visualization files to save
        batch_size: Number of documents to process before saving batch metadata
        resume_from: Document index to resume from (if continuing a previous run)
    """
    print(f"Creating {num_documents} documents starting from index {resume_from}...")
    
    # Check for progress file to resume from
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")
    batch_labels = []
    all_labels_count = 0
    
    # If resuming, load progress information
    if resume_from > 0 and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            all_labels_count = progress.get("total_documents", 0)
            print(f"Resuming from document {resume_from}, already completed {all_labels_count} documents")
    
    # Create a directory for LaTeX files
    latex_dir = os.path.join(OUTPUT_DIR, "latex")
    os.makedirs(latex_dir, exist_ok=True)
    
    # Create directory for batch metadata
    batch_dir = os.path.join(OUTPUT_DIR, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Process documents in batches
    for i in tqdm(range(resume_from, num_documents)):
        # Generate document ID
        doc_id = f"doc_{i:04d}"
        
        # Create document with more elements to fill the page
        doc_image, annotation_image, doc_labels = create_document(
            iam_metadata, math_metadata, doc_id, num_elements=40
        )
        
        # Save document image
        doc_path = os.path.join(OUTPUT_DIR, "images", f"{doc_id}.jpg")
        # Convert to grayscale before saving to reduce file size
        gray_image = cv2.cvtColor(doc_image, cv2.COLOR_RGB2GRAY)
        if resize_factor != 1.0:
            new_width = int(gray_image.shape[1] * resize_factor)
            new_height = int(gray_image.shape[0] * resize_factor)
            gray_image = cv2.resize(gray_image, (new_width, new_height), 
                                  interpolation=cv2.INTER_AREA)
                                  
        cv2.imwrite(doc_path, gray_image, [cv2.IMWRITE_JPEG_QUALITY, 65])
        
        # Save visualization annotation only for the first max_visualizations documents
        if i < max_visualizations:
            annotation_path = os.path.join(OUTPUT_DIR, "visualizations", f"{doc_id}_annotated.jpg")
            color_anno = cv2.cvtColor(annotation_image, cv2.COLOR_RGB2BGR)
            
            # Resize annotation image
            if resize_factor != 1.0:
                new_width = int(color_anno.shape[1] * resize_factor)
                new_height = int(color_anno.shape[0] * resize_factor)
                color_anno = cv2.resize(color_anno, (new_width, new_height),
                                      interpolation=cv2.INTER_AREA)
                                      
            cv2.imwrite(annotation_path, color_anno, [cv2.IMWRITE_JPEG_QUALITY, 65])
            print(f"Saved visualization for document {doc_id}")
        
        # Save labels as JSON
        label_path = os.path.join(OUTPUT_DIR, "annotations", f"{doc_id}.json")
        with open(label_path, 'w') as f:
            json.dump(doc_labels, f, indent=2)
        
        # Save LaTeX document as a separate .tex file
        latex_path = os.path.join(latex_dir, f"{doc_id}.tex")
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(doc_labels["latex_document"])
        
        # Add to batch labels
        batch_labels.append(doc_labels)
        all_labels_count += 1
        
        # Process batch if we've reached batch_size or the last document
        current_batch = (i // batch_size)
        is_last_document = (i == num_documents - 1)
        is_batch_complete = (i % batch_size == batch_size - 1) or is_last_document
        
        if is_batch_complete:
            # Create a CSV for this batch of data
            batch_rows = []
            for doc in batch_labels:
                doc_id = doc["document_id"]
                for elem in doc["elements"]:
                    # Create row with all element data including LaTeX and alignment
                    row = {
                        "document_id": doc_id,
                        "element_id": elem["id"],
                        "type": elem["type"],
                        "text": elem["text"],
                        "latex": elem["latex"],
                        "alignment_type": elem["alignment"],
                        "x1": elem["bbox"][0],
                        "y1": elem["bbox"][1],
                        "x2": elem["bbox"][2],
                        "y2": elem["bbox"][3],
                        "source": elem["original_source"]
                    }
                    
                    # Add detailed alignment information
                    if elem["alignment"] == "random_shift":
                        margin_left = 120  # Same as in create_document function
                        shift_pixels = elem["bbox"][0] - margin_left
                        shift_ems = shift_pixels / 10  # Convert pixels to em units
                        row["alignment_details"] = f"shift_by_{shift_ems:.2f}em"
                        row["latex_alignment"] = f"\\noindent\\hspace{{{shift_ems:.2f}em}}"
                    elif elem["alignment"] == "center":
                        row["alignment_details"] = "centered"
                        row["latex_alignment"] = "\\begin{center}...\\end{center}"
                    else:  # left alignment
                        row["alignment_details"] = "no_indent"
                        row["latex_alignment"] = "\\noindent"
                    
                    # Add symbol-specific fields if applicable
                    if "has_associated_symbol" in elem and elem["has_associated_symbol"]:
                        row["has_symbol"] = True
                        row["symbol_id"] = elem["associated_symbol_id"]
                    elif "associated_text_id" in elem:
                        row["is_symbol"] = True
                        row["associated_text_id"] = elem["associated_text_id"]
                    
                    batch_rows.append(row)
            
            # Save batch metadata
            batch_df = pd.DataFrame(batch_rows)
            batch_csv_path = os.path.join(batch_dir, f"batch_{current_batch}.csv")
            batch_df.to_csv(batch_csv_path, index=False)
            
            # Save batch labels
            batch_json_path = os.path.join(batch_dir, f"batch_{current_batch}.json")
            with open(batch_json_path, 'w') as f:
                json.dump({
                    "batch": current_batch,
                    "start_doc": current_batch * batch_size,
                    "end_doc": i,
                    "documents": batch_labels
                }, f)
            
            # Update progress file
            with open(progress_file, 'w') as f:
                json.dump({
                    "last_completed": i,
                    "current_batch": current_batch,
                    "total_documents": all_labels_count
                }, f)
            
            print(f"Completed batch {current_batch} ({len(batch_labels)} documents)")
            print(f"Total progress: {all_labels_count}/{num_documents} documents ({all_labels_count/num_documents*100:.1f}%)")
            
            # Clear batch labels to free memory
            batch_labels = []
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
    
    # After all batches are complete, merge CSVs if needed
    print("Processing complete. Merging batch CSVs...")
    all_batches = []
    for batch_file in sorted(os.listdir(batch_dir)):
        if batch_file.endswith('.csv'):
            batch_path = os.path.join(batch_dir, batch_file)
            batch_df = pd.read_csv(batch_path)
            all_batches.append(batch_df)
    
    # Combine all batch CSVs
    if all_batches:
        full_df = pd.concat(all_batches, ignore_index=True)
        full_df.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"), index=False)
        print(f"Combined {len(all_batches)} batches into final dataset.csv")
    
    # Create combined dataset.json (optional, might be very large)
    # Instead of loading all batch JSONs into memory, just create a summary file
    summary = {
        "dataset": "Combined IAM and MathWriting",
        "total_documents": all_labels_count,
        "total_batches": current_batch + 1,
        "batch_size": batch_size,
        "creation_complete": True
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Created {all_labels_count} documents across {current_batch + 1} batches")
    print(f"Dataset saved to {OUTPUT_DIR}")

def display_sample_documents(num_samples=2):
    """Display sample documents with their annotations"""
    # Get list of document IDs
    doc_ids = [f.split('.')[0] for f in os.listdir(os.path.join(OUTPUT_DIR, "images")) 
          if f.endswith('.jpg')]
    
    if not doc_ids:
        print("No documents found.")
        return
    
    # Select random samples
    samples = random.sample(doc_ids, min(num_samples, len(doc_ids)))
    
    # Display each sample
    for doc_id in samples:
        # Load images
        doc_path = os.path.join(OUTPUT_DIR, "images", f"{doc_id}.jpg")
        annotation_path = os.path.join(OUTPUT_DIR, "visualizations", f"{doc_id}_annotated.jpg")
        
        doc_img = cv2.imread(doc_path)
        if len(doc_img.shape) == 3:  # If it's color
            doc_img = cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB)
        
        anno_img = cv2.imread(annotation_path)
        anno_img = cv2.cvtColor(anno_img, cv2.COLOR_BGR2RGB)
        
        # Load labels
        with open(os.path.join(OUTPUT_DIR, "annotations", f"{doc_id}.json"), 'r') as f:
            labels = json.load(f)
        
        # Display
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(doc_img)
        axes[0].set_title(f"Document: {doc_id}")
        axes[0].axis('off')
        
        axes[1].imshow(anno_img)
        axes[1].set_title(f"Annotations: {len(labels['elements'])} elements")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print element info
        print(f"\nDocument {doc_id} elements:")
        for i, elem in enumerate(labels["elements"]):
            print(f"{i+1}. Type: {elem['type']}, Text: {elem['text'][:30]}{'...' if len(elem['text']) > 30 else ''}")

# Main execution
if __name__ == "__main__":
    # Check that directories exist
    for dir_path in [IAM_DIR, MATHWRITING_DIR]:
        if not os.path.exists(dir_path):
            print(f"Error: Directory not found: {dir_path}")
            print("Please check the path and try again.")
            exit(1)
    
    # Verify IAM subfolder structure
    for subfolder in ['train', 'test', 'validation']:
        folder_path = os.path.join(IAM_DIR, subfolder)
        if not os.path.exists(folder_path):
            print(f"Warning: IAM subfolder not found: {folder_path}")
    
    # Check for train folder in MathWriting directory
    train_dir = os.path.join(MATHWRITING_DIR, "train")
    if not os.path.exists(train_dir):
        print(f"Error: Train directory not found: {train_dir}")
        print("Please check that the 'train' folder exists in the MathWriting directory.")
        exit(1)
    
    print(f"Using math equations ONLY from: {train_dir}")
    
    # Load datasets
    iam_metadata, math_metadata = load_datasets()
    
    # Check if we have valid data
    if len(iam_metadata) == 0:
        print("Error: No valid IAM text lines found. Cannot create document.")
        exit(1)
    
    if len(math_metadata) == 0:
        print("Error: No valid math equations found in train folder. Please check the dataset.")
        exit(1)
    else:
        print(f"Successfully loaded {len(math_metadata)} math equations from train folder")
    
    # Debug: Display dataset information
    print(f"\nIAM dataset: {len(iam_metadata)} entries")
    print(f"Math dataset: {len(math_metadata)} entries")
    
    # Create dataset
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")
    resume_from = 0
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                resume_from = progress.get("last_completed", -1) + 1
                print(f"Found existing progress. Resume from document {resume_from}? (y/n)")
                response = input().lower()
                if response != 'y':
                    resume_from = 0
                    print("Starting from the beginning.")
                else:
                    print(f"Resuming from document {resume_from}.")
        except:
            print("Could not read progress file. Starting from the beginning.")
    
    # Create dataset with batch saving
    create_dataset(
        iam_metadata, 
        math_metadata, 
        num_documents=50000, 
        resize_factor=1, 
        max_visualizations=5,
        batch_size=1000,  # Adjust this based on your available RAM
        resume_from=resume_from
    )
    # Display samples
    #display_sample_documents(2)