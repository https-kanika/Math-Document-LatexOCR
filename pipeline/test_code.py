import cv2
import json
import numpy as np
import os
from pathlib import Path

def restitch_document(metadata_path):
    """
    Restitch the original document from segmented images using metadata.
    
    Args:
        metadata_path: Path to the metadata.json file
        
    Returns:
        restitched_image: The reconstructed image
    """
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get original image dimensions
    img_width = metadata['original_image']['width']
    img_height = metadata['original_image']['height']
    
    # Create blank canvas
    canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # White background
    
    # Get the folder containing the segments
    output_folder = os.path.dirname(metadata_path)
    
    # Place each segment back on the canvas
    for segment in metadata['segments']:
        segment_path = os.path.join(output_folder, segment['filename'])
        
        if not os.path.exists(segment_path):
            print(f"Warning: {segment['filename']} not found, skipping...")
            continue
        
        # Read the segment image
        segment_img = cv2.imread(segment_path)
        
        # Get coordinates
        x1 = segment['bbox']['x1']
        y1 = segment['bbox']['y1']
        x2 = segment['bbox']['x2']
        y2 = segment['bbox']['y2']
        
        # Place segment on canvas at original position
        canvas[y1:y2, x1:x2] = segment_img
        
        print(f"Placed segment {segment['id']} ({segment['class']}) at position ({x1}, {y1})")
    
    return canvas

def restitch_and_save(metadata_path, output_filename='restitched_image.jpg'):
    """
    Restitch document and save it
    
    Args:
        metadata_path: Path to the metadata.json file
        output_filename: Name of the output file
    """
    # Restitch the image
    restitched = restitch_document(metadata_path)
    
    # Save in the same folder as metadata
    output_folder = os.path.dirname(metadata_path)
    output_path = os.path.join(output_folder, output_filename)
    
    cv2.imwrite(output_path, restitched)
    print(f"\nRestitched image saved to: {output_path}")
    
    return output_path

# Test the function
if __name__ == "__main__":
    # Example usage - replace with your actual metadata path
    metadata_path = r"C:\Users\kani1\Desktop\IE643\pipeline\outputs\506f08fa-4376-41b6-8a6d-a3121e0bc275\metadata.json"
    
    if os.path.exists(metadata_path):
        output_path = restitch_and_save(metadata_path)
        print(f"Success! Restitched image at: {output_path}")
    else:
        print(f"Metadata file not found at: {metadata_path}")
        print("\nTo use this script:")
        print("1. Run the Flask app and upload an image")
        print("2. Copy the request_id from the response")
        print("3. Update metadata_path with: outputs\\<request_id>\\metadata.json")