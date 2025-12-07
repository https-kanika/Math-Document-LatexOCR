import torch
import pickle
import sys
from pathlib import Path
import cv2
import numpy as np
import torch.nn.functional as F

# Add the math-detection folder to path
current_dir = Path(__file__).resolve().parent  # pipeline folder
math_detection_dir = current_dir.parent / "math-detection"  # Go up one level, then into math-detection
sys.path.insert(0, str(math_detection_dir))

print(f"Added to sys.path: {math_detection_dir}")  # Debug print

from model_final import FullyConvolutionalNetwork, GRUDecoder, reshape_fcn_output


def make_5ch_from_image_path(image_path, out_size=(800, 240), blur_sigma=1.0, thick_radius=1, device="cuda"):
    """
    Convert image to 5-channel tensor for model input
    Returns: torch.Tensor shape (5, H, W)
    """
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
    dirs = torch.cat(dirs, dim=1)  # (1, 4, H, W)

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
        kernel_x = kernel.view(1, 1, -1, 1).repeat(dirs.shape[1], 1, 1, 1)
        kernel_y = kernel.view(1, 1, 1, -1).repeat(dirs.shape[1], 1, 1, 1)

        dirs = F.conv2d(dirs, kernel_x, padding=(radius, 0), groups=dirs.shape[1])
        dirs = F.conv2d(dirs, kernel_y, padding=(0, radius), groups=dirs.shape[1])

    # Normalize
    dirs = torch.sqrt(dirs / (dirs.amax(dim=(2, 3), keepdim=True) + 1e-12))

    # Stack grayscale + directional: (1, 1, H, W) + (1, 4, H, W) = (1, 5, H, W)
    five = torch.cat([gray, dirs], dim=1)  # (1, 5, H, W)
    
    return five.squeeze(0)  # (5, H, W)


def load_wap_model(checkpoint_path, word2idx_path, idx2word_path, device='cuda'):
    """Load trained WAP model"""
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


def transcribe_image(image_path, encoder, decoder, word2idx, idx2word, device, 
                     beam_width=10, max_len=150):
    """
    Transcribe math equation/symbol to LaTeX
    
    Args:
        image_path: Path to input image
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
        )  # Shape: (5, H, W)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # (1, 5, H, W)
        
        # Get special token indices
        start_token = word2idx['<START>']
        end_token = word2idx['<END>']
        
        with torch.no_grad():
            # Encode image
            encoder_output = encoder(image_tensor)
            annotations = reshape_fcn_output(encoder_output)
            
            # Decode using beam search
            predicted_sequence, attention_weights = decoder.decode_beam_search(
                annotations=annotations,
                start_token=start_token,
                end_token=end_token,
                max_len=max_len,
                beam_width=beam_width
            )
        
        # Convert indices to tokens
        latex_tokens = []
        for idx in predicted_sequence:
            if idx == start_token:
                continue
            if idx == end_token:
                break
            token = idx2word.get(idx, '<UNK>')
            latex_tokens.append(token)
        
        # Join tokens to form LaTeX string
        latex_string = ' '.join(latex_tokens)
        
        return latex_string
    
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return None


def main():
    # Configuration
    CHECKPOINT_PATH = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\math-detection\checkpoint_annealed_best.pth"
    WORD2IDX_PATH = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\pipeline\vocab\word2idx.pkl"
    IDX2WORD_PATH = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\pipeline\vocab\idx2word.pkl"
    
    # Input image path (change this to your image)
    IMAGE_PATH = r"C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessedCrome2014Data\train\65_alfonso.png"
    
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"Error: Image not found at {IMAGE_PATH}")
        print("Please update IMAGE_PATH in the script to point to your image.")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading WAP model...")
    encoder, decoder, word2idx, idx2word, device = load_wap_model(
        checkpoint_path=CHECKPOINT_PATH,
        word2idx_path=WORD2IDX_PATH,
        idx2word_path=IDX2WORD_PATH,
        device=device
    )
    print("Model loaded successfully!")
    
    # Transcribe image
    print(f"\nTranscribing image: {IMAGE_PATH}")
    latex_output = transcribe_image(
        image_path=IMAGE_PATH,
        encoder=encoder,
        decoder=decoder,
        word2idx=word2idx,
        idx2word=idx2word,
        device=device,
        beam_width=10,
        max_len=150
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULT:")
    print("="*80)
    if latex_output:
        print(f"LaTeX: {latex_output}")
    else:
        print("Failed to transcribe image")
    print("="*80)


if __name__ == '__main__':
    main()