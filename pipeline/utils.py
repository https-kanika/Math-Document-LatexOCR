import os
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from math_model.model_mumz import FullyConvolutionalNetwork, GRUDecoder, reshape_fcn_output
import torch
import pickle
import json
import os
from pathlib import Path

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


# Load your trained model (do this once at module level)
def load_latex_model(checkpoint_path, word2idx_path, idx2word_path, device='cuda'):
    """
    Load the trained WAP model for LaTeX transcription
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        word2idx_path: Path to word2idx vocabulary pickle
        idx2word_path: Path to idx2word vocabulary pickle
        device: Device to load model on
        
    Returns:
        encoder, decoder, word2idx, idx2word, device
    """
    # Load vocabularies
    with open(word2idx_path, 'rb') as f:
        word2idx = pickle.load(f)
    with open(idx2word_path, 'rb') as f:
        idx2word = pickle.load(f)
    
    VOCAB_SIZE = len(set(word2idx.values()))
    EMBEDDING_DIM = 256
    DECODER_DIM = 256
    ENCODER_DIM = 128
    ATTENTION_DIM = 512
    COVERAGE_KERNEL_SIZE = 11
    
    # Initialize models
    encoder = FullyConvolutionalNetwork()
    decoder = GRUDecoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        decoder_dim=DECODER_DIM,
        encoder_dim=ENCODER_DIM,
        attention_dim=ATTENTION_DIM,
        kernel_size=COVERAGE_KERNEL_SIZE
    )
    
    # Load checkpoint
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, word2idx, idx2word, device



# ...existing code...

def make_5ch_from_image_path(image_path, out_size=(800, 240), blur_sigma=1.0, thick_radius=1, device="cuda"):
    """
    Convert image to 5-channel tensor for model input
    Same as make_5ch_from_image_gpu but takes image path
    
    Returns: torch.Tensor shape (5, H, W)
    """
    import cv2
    import numpy as np
    import torch.nn.functional as F
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")

    img = img.astype(np.float32) / 255.0
    if out_size is not None:
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_LINEAR)

    img_t = torch.from_numpy(img).to(device)
    gray = img_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Sobel filters
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(2, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    ori = torch.atan2(gy, gx)

    # Directional bins: 4 directions
    nbins = 4
    bin_edges = torch.linspace(-np.pi, np.pi, nbins + 1, device=device)
    dirs = []
    for b in range(nbins):
        mask = ((ori >= bin_edges[b]) & (ori < bin_edges[b + 1])).float()
        dirs.append(mag * mask)
    dirs = torch.cat(dirs, dim=0)  # (4, 1, H, W)

    # Thickening
    if thick_radius > 0:
        k = 2 * thick_radius + 1
        dirs = F.max_pool2d(dirs, kernel_size=k, stride=1, padding=thick_radius)

    # Gaussian blur
    if blur_sigma > 0:
        radius = int(3 * blur_sigma)
        x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / blur_sigma) ** 2)
        kernel /= kernel.sum()
        kernel_x = kernel.view(1, 1, -1, 1).repeat(dirs.shape[0], 1, 1, 1)
        kernel_y = kernel.view(1, 1, 1, -1).repeat(dirs.shape[0], 1, 1, 1)

        dirs = dirs.permute(1, 0, 2, 3)  # (1, 4, H, W)
        dirs = F.conv2d(dirs, kernel_x, padding=(radius, 0), groups=dirs.shape[1])
        dirs = F.conv2d(dirs, kernel_y, padding=(0, radius), groups=dirs.shape[1])
        dirs = dirs.permute(1, 0, 2, 3)  # (4, 1, H, W)

    # Normalize
    dirs = torch.sqrt(dirs / (dirs.amax(dim=(2, 3), keepdim=True) + 1e-12))

    # Stack grayscale + directional: (1, 1, H, W) + (4, 1, H, W) = (5, 1, H, W)
    five = torch.cat([gray, dirs], dim=0)  # (5, 1, H, W)
    
    # FIXED: Remove the extra dimension to get (5, H, W)
    return five.squeeze(1)  # Changed from squeeze(0) to squeeze(1)


# ...existing code...

def transcribe_latex_with_wap(image_path, encoder, decoder, word2idx, idx2word, device, 
                               beam_width=10, max_len=150):
    """
    Transcribe math equation/symbol to LaTeX using trained WAP model
    
    Args:
        image_path: Path to the image file
        encoder: Trained encoder model
        decoder: Trained decoder model
        word2idx: Vocabulary mapping
        idx2word: Reverse vocabulary mapping
        device: torch device
        beam_width: Beam search width
        max_len: Maximum sequence length
        
    Returns:
        LaTeX string
    """
    try:
        # Convert image to 5-channel tensor
        image_tensor = make_5ch_from_image_path(
            image_path,
            out_size=(800, 240),
            blur_sigma=1.0,
            thick_radius=1,
            device=device
        )  # Shape: (5, H, W) where H=240, W=800
        
        # Add batch dimension: (1, 5, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # Now it's (1, 5, 240, 800)
        
        # Get special token indices
        start_token = word2idx['<START>']
        end_token = word2idx['<END>']
        
        with torch.no_grad():
            # Encode image
            encoder_output = encoder(image_tensor)  # (1, 128, H', W')
            annotations = reshape_fcn_output(encoder_output)  # (1, L, 128)
            
            # Decode using beam search - returns (sequence_list, attention_list)
            predicted_sequence, attention_weights = decoder.decode_beam_search(
                annotations=annotations,
                start_token=start_token,
                end_token=end_token,
                max_len=max_len,
                beam_width=beam_width
            )
        
        # predicted_sequence is already a list of indices
        # Convert indices to tokens, skipping special tokens
        latex_tokens = []
        for idx in predicted_sequence:
            if idx == start_token:
                continue
            if idx == end_token:
                break
            token = idx2word.get(idx, '<UNK>')
            latex_tokens.append(token)
        
        # Join tokens to form LaTeX string (with spaces)
        latex_string = ' '.join(latex_tokens)
        
        return latex_string
    
    except Exception as e:
        import traceback
        error_msg = f"Error transcribing LaTeX: {str(e)}"
        print(f"\n{error_msg}")
        print(f"Image path: {image_path}")
        print(traceback.format_exc())  # Print full traceback for debugging
        return error_msg

def escape_latex_text(s: str) -> str:
    """
    Escape characters that have special meaning in LaTeX for plain text segments.
    """
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def sanitize_segment_transcription(transcription: str, seg_class: str) -> str:
    """
    Prepare LaTeX-friendly content for a segment transcription.
    - For equations and symbols, assume the transcription is already LaTeX-like.
    - For text, escape problematic characters.
    """
    if transcription is None:
        return ""
    
    if seg_class in ["equation", "symbol"]:
        return transcription.strip()
    else:  # text
        return escape_latex_text(transcription.strip())

def build_latex_document(segments, original_dims=None):
    """
    Build a LaTeX document string from segments based on their positions.
    Segments are sorted by vertical position (y1) first, then horizontal (x1).
    This recreates the document flow from top to bottom, left to right.
    """
    # Sort segments by position: top to bottom (y1), then left to right (x1)
    sorted_segments = sorted(segments, key=lambda s: (s['bbox']['y1'], s['bbox']['x1']))
    
    body_lines = []
    
    for seg in sorted_segments:
        cls = seg.get("class", "")
        transcription = seg.get("transcription", "")
        
        # Skip if no transcription
        if not transcription or transcription.startswith("Error"):
            continue
        
        content = sanitize_segment_transcription(transcription, cls)
        
        if cls == "equation":
            # Display equation on its own line
            body_lines.append("\n\\[\n" + content + "\n\\]\n")
        elif cls == "symbol":
            # Inline math for symbols
            body_lines.append("$" + content + "$")
        else:  # text
            # Regular text
            body_lines.append(content)
        
        # Add spacing between segments
        body_lines.append("\n\n")
    
    # Assemble full LaTeX document
    header = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

\begin{document}

"""
    
    footer = r"""
\end{document}
"""
    
    body = "".join(body_lines)
    
    return header + body + footer

def read_metadata_and_generate_tex(metadata_path, output_tex_path):
    """
    Read the given metadata.json file and generate a LaTeX .tex file at output_tex_path.
    
    Args:
        metadata_path: Path to metadata.json file
        output_tex_path: Path where the .tex file should be saved
        
    Returns:
        Path to the generated .tex file
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Read metadata
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get segments
    segments = data.get("segments", [])
    
    if not segments:
        raise ValueError("No segments found in metadata")
    
    # Get original image dimensions (optional, for reference)
    original_dims = data.get("original_image", {})
    
    # Filter segments that have valid transcriptions
    valid_segments = []
    for seg in segments:
        if "transcription" in seg and seg["transcription"] and not seg["transcription"].startswith("Error"):
            valid_segments.append(seg)
    
    if not valid_segments:
        raise ValueError("No valid transcriptions found in segments")
    
    # Build LaTeX document
    latex_content = build_latex_document(valid_segments, original_dims)
    
    # Write to output file
    output_tex_path = Path(output_tex_path)
    output_tex_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_tex_path.open("w", encoding="utf-8") as f:
        f.write(latex_content)
    
    print(f"✓ LaTeX document generated: {output_tex_path}")
    print(f"  Total segments: {len(segments)}")
    print(f"  Valid transcriptions: {len(valid_segments)}")
    
    return str(output_tex_path)
