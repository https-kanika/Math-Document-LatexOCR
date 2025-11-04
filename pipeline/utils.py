import os
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def apply_binarization(image_path, save_path=None, method='otsu', threshold=127, 
                      adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block_size=11, C=2):
    """
    Apply binarization to convert grayscale image to binary image
    
    Args:
        image_path: Path to the input image
        save_path: Path to save the processed image (optional)
        method: Binarization method ('simple', 'otsu', 'adaptive')
        threshold: Threshold value for simple thresholding (0-255)
        adaptive_method: Method for adaptive thresholding
        block_size: Size of pixel neighborhood for adaptive threshold (must be odd)
        C: Constant subtracted from mean or weighted mean (adaptive threshold)
        
    Returns:
        The binarized image
    """
    # Check if file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Try different methods to read the image
    try:
        # Read with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("OpenCV couldn't read the image")
    except Exception as e:
        try:
            # Fallback to PIL
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
            # Convert RGB to BGR for OpenCV if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e2:
            raise ValueError(f"Could not read image with OpenCV or PIL: {str(e)} | {str(e2)}")
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply binarization based on specified method
    if method == 'simple':
        # Simple thresholding
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif method == 'otsu':
        # Otsu's thresholding (automatically determines optimal threshold)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, adaptive_method, 
                                      cv2.THRESH_BINARY, block_size, C)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    
    # Save the binarized image if a save path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, binary)
    
    return binary


def transcribe_text_with_trocr(image_path):
    """
    Transcribe text from an image using TrOCR
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Transcribed text string
    """
    try:
        # Load and convert image
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        pixel_values = trocr_processor(image, return_tensors="pt").pixel_values
        
        # Generate transcription
        generated_ids = trocr_model.generate(pixel_values)
        transcribed_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return transcribed_text
    
    except Exception as e:
        return f"Error transcribing: {str(e)}"
