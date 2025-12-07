import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import pickle
import json
import re

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_latex_label(latex_string):
    """Remove $ signs and style commands from LaTeX"""
    cleaned = latex_string.strip()
    
    if cleaned.startswith('$'):
        cleaned = cleaned[1:]
    if cleaned.endswith('$'):
        cleaned = cleaned[:-1]
    
    # Remove style commands
    style_commands = [r'\mbox', r'\hbox', r'\mathrm', r'\vtop']
    for cmd in style_commands:
        while cmd in cleaned:
            pattern = re.escape(cmd) + r'\s*\{'
            match = re.search(pattern, cleaned)
            if not match:
                break
            
            start = match.start()
            brace_start = match.end() - 1
            
            brace_count = 1
            i = brace_start + 1
            while i < len(cleaned) and brace_count > 0:
                if cleaned[i] == '{':
                    brace_count += 1
                elif cleaned[i] == '}':
                    brace_count -= 1
                i += 1
            
            if brace_count == 0:
                content = cleaned[brace_start + 1:i - 1]
                cleaned = cleaned[:start] + '{' + content + '}' + cleaned[i:]
            else:
                cleaned = cleaned[:start] + cleaned[match.end():]
    
    # Remove delimiter sizing commands
    delimiter_commands = [
        r'\\Bigg\s*', r'\\bigg\s*', r'\\Big\s*', r'\\big\s*',
        r'\\left\s*', r'\\right\s*', r'\\limits\s*'
    ]
    for pattern in delimiter_commands:
        cleaned = re.sub(pattern, '', cleaned)
    
    return cleaned.strip()


def load_vocabulary(vocab_dir='./vocab/'):
    """Load vocabulary from saved files"""
    with open(os.path.join(vocab_dir, 'token2idx.pkl'), 'rb') as f:
        token2idx = pickle.load(f)
    
    with open(os.path.join(vocab_dir, 'idx2token.pkl'), 'rb') as f:
        idx2token = pickle.load(f)
    
    return token2idx, idx2token


class LaTeXTokenizer:
    """Custom tokenizer for LaTeX expressions"""
    
    def __init__(self, token2idx, idx2token):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.vocab_size = len(token2idx)
        
        # Special token IDs
        self.pad_token_id = token2idx['<PAD>']
        self.cls_token_id = token2idx['<START>']
        self.sep_token_id = token2idx['<END>']
        self.unk_token_id = token2idx['<UNK>']
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs to text"""
        tokens = []
        for idx in token_ids:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if skip_special_tokens and token in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)


def find_best_checkpoint(checkpoint_dir, exclude_latest=True):
    """Find checkpoint with lowest validation loss"""
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    
    if exclude_latest and len(checkpoints) > 1:
        checkpoints = checkpoints[:-1]
    
    best_checkpoint = None
    best_loss = float('inf')
    
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
        
        if os.path.exists(trainer_state_file):
            try:
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                
                eval_losses = [log['eval_loss'] for log in trainer_state.get('log_history', []) 
                              if 'eval_loss' in log]
                
                if eval_losses:
                    current_loss = eval_losses[-1]
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_checkpoint = checkpoint_path
            except Exception as e:
                continue
    
    if best_checkpoint is None:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        return checkpoint_path
    
    print(f"Best checkpoint: {os.path.basename(best_checkpoint)} (eval_loss = {best_loss:.4f})")
    return best_checkpoint


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class LaTeXOCRInference:
    """Inference class for LaTeX OCR model"""
    
    def __init__(self, checkpoint_path=None, vocab_dir=r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\math-detection\fine-tune-trocr-crohme14\vocab", device=None):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to model checkpoint (if None, finds best)
            vocab_dir: Directory containing vocabulary files
            device: torch device (if None, auto-detects)
        """
        # Load vocabulary
        print("Loading vocabulary...")
        token2idx, idx2token = load_vocabulary(vocab_dir)
        self.tokenizer = LaTeXTokenizer(token2idx, idx2token)
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        
        # Find best checkpoint if not specified
        if checkpoint_path is None:
            print("\nFinding best checkpoint...")
            checkpoint_path= r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\math-detection\fine-tune-trocr-crohme14"
        
        print(f"Loading model from: {checkpoint_path}")
        
        # ✅ FIX: Load image processor from saved checkpoint, not from HuggingFace
        try:
            # Try loading from your saved model first
            from transformers import ViTImageProcessor
            self.image_processor = ViTImageProcessor.from_pretrained(checkpoint_path)
            print("Loaded image processor from saved model")
        except:
            # Fallback: Load from HuggingFace (for initial TrOCR)
            trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
            self.image_processor = trocr_processor.image_processor
            print("Loaded image processor from HuggingFace")
        
        # Load model
        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
        self.model.eval()
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}\n")
    
    def predict(self, image_path, return_raw=False):
        """
        Run inference on a single image
        
        Args:
            image_path: Path to image file or PIL Image
            return_raw: If True, returns raw token IDs along with text
        
        Returns:
            LaTeX string (or tuple of (latex_string, token_ids) if return_raw=True)
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")
        
        # Process image
        pixel_values = self.image_processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode
        latex_string = self.tokenizer.decode(
            generated_ids[0].tolist(), 
            skip_special_tokens=True
        )
        
        if return_raw:
            return latex_string, generated_ids[0].tolist()
        
        return latex_string
    
    def predict_batch(self, image_paths, batch_size=8):
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image paths or PIL Images
            batch_size: Batch size for processing
        
        Returns:
            List of LaTeX strings
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Load and process images
            images = []
            for path in batch_paths:
                if isinstance(path, str):
                    img = Image.open(path).convert("RGB")
                else:
                    img = path.convert("RGB")
                images.append(img)
            
            pixel_values = self.image_processor(
                images=images,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode batch
            batch_results = self.tokenizer.decode(
                generated_ids.tolist(),
                skip_special_tokens=True
            )
            
            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize inference model
    ocr = LaTeXOCRInference()
    
    # Example 1: Single image prediction
    print("="*50)
    print("SINGLE IMAGE INFERENCE")
    print("="*50)
    
    image_path = r"C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessedCrome2014Data\train\65_herbert.png"  # Update with your image path
    
    if os.path.exists(image_path):
        latex_output = ocr.predict(image_path)
        print(f"Image: {image_path}")
        print(f"Predicted LaTeX: {latex_output}")
        print()
    
    # Example 2: Single image with raw token IDs
    print("="*50)
    print("WITH RAW TOKEN IDs")
    print("="*50)
    
    if os.path.exists(image_path):
        latex_output, token_ids = ocr.predict(image_path, return_raw=True)
        print(f"Predicted LaTeX: {latex_output}")
        print(f"Token IDs (first 20): {token_ids[:20]}")
        print()
    
    # # Example 3: Batch prediction
    # print("="*50)
    # print("BATCH INFERENCE")
    # print("="*50)
    
    # # Get first 5 images from test set
    # test_images = [
    #     f"./data/2014/{filename}" 
    #     for filename in os.listdir("./data/2014/")[:5]
    #     if filename.endswith('.jpg')
    # ]
    
    # if test_images:
    #     predictions = ocr.predict_batch(test_images, batch_size=2)
        
    #     for img_path, pred in zip(test_images, predictions):
    #         print(f"{os.path.basename(img_path)}: {pred}")
    
    # print("\n" + "="*50)
    # print("Inference complete!")
    # print("="*50)