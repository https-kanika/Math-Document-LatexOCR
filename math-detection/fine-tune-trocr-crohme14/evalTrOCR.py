import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
import pandas as pd
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import torch
import evaluate
from tqdm import tqdm
import json
import pickle
import re
from collections import Counter

# ============================================================================
# COPY THESE FUNCTIONS FROM trainVocab.py
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


def tokenize_latex(latex_string):
    """Tokenize LaTeX string into individual tokens"""
    tokens = []
    i = 0
    
    while i < len(latex_string):
        if latex_string[i] == '\\':
            j = i + 1
            while j < len(latex_string) and latex_string[j].isalpha():
                j += 1
            
            if j > i + 1:
                tokens.append(latex_string[i:j])
                i = j
            else:
                if j < len(latex_string):
                    tokens.append(latex_string[i:j+1])
                    i = j + 1
                else:
                    tokens.append(latex_string[i])
                    i += 1
        elif latex_string[i].isspace():
            i += 1
        elif latex_string[i] in '{}[]()^_=+-*/|<>!.,:;':
            tokens.append(latex_string[i])
            i += 1
        else:
            tokens.append(latex_string[i])
            i += 1
    
    return tokens


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
    
    def batch_decode(self, sequences, skip_special_tokens=True):
        """Batch decode sequences"""
        return [self.decode(seq, skip_special_tokens) for seq in sequences]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def find_best_checkpoint(checkpoint_dir, exclude_latest=True):
    """Find checkpoint with lowest validation loss"""
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    
    if exclude_latest and len(checkpoints) > 1:
        checkpoints = checkpoints[:-1]
        print(f"Excluding latest checkpoint: checkpoint-{checkpoints[-1].split('-')[-1]}")
    
    best_checkpoint = None
    best_loss = float('inf')
    
    print("\nSearching for best checkpoint based on validation loss...\n")
    
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
                    print(f"{checkpoint}: eval_loss = {current_loss:.4f}")
                    
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_checkpoint = checkpoint_path
            except Exception as e:
                print(f"Could not read {checkpoint}: {e}")
    
    if best_checkpoint is None:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"\nNo eval loss found, using: {checkpoints[-1]}")
        return checkpoint_path
    
    print(f"\nBest checkpoint: {os.path.basename(best_checkpoint)} with eval_loss = {best_loss:.4f}\n")
    return best_checkpoint


def evaluate_model(model, image_processor, latex_tokenizer, df, root_dir, device):
    """Evaluate model with custom vocabulary"""
    predictions = []
    references = []
    
    for idx in tqdm(range(len(df))):
        file_name = df['file_name'].iloc[idx]
        text = df['text'].iloc[idx]
        
        # Load and process image
        image = Image.open(root_dir + file_name).convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        # DEBUG: Print raw token IDs
        if idx < 5:
            print(f"\n=== Example {idx+1} DEBUG ===")
            print(f"Generated IDs: {generated_ids[0].tolist()[:20]}")  # First 20 tokens
            print(f"Model vocab size: {model.config.vocab_size}")
            print(f"Tokenizer vocab size: {latex_tokenizer.vocab_size}")
        
        # Decode using custom tokenizer - DON'T skip special tokens initially
        pred_str_with_special = latex_tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
        pred_str = latex_tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        
        # Clean the reference text the same way
        ref_str = clean_latex_label(str(text))
        
        predictions.append(pred_str)
        references.append(ref_str)
        
        # Print first 5 examples for debugging
        if idx < 5:
            print(f"With special tokens: {pred_str_with_special}")
            print(f"Without special tokens: {pred_str}")
            print(f"Ground truth: {ref_str}")
            print(f"Match: {'✓' if pred_str == ref_str else '✗'}")
    
    
    # Compute metrics
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Calculate exact match accuracy
    exact_matches = sum([1 for pred, ref in zip(predictions, references) if pred == ref])
    accuracy = exact_matches / len(predictions)
    
    return {
        "cer": cer,
        "wer": wer,
        "accuracy": accuracy,
        "total_samples": len(predictions),
        "exact_matches": exact_matches
    }


# ============================================================================
# MAIN EVALUATION SCRIPT
# ============================================================================

# Load vocabulary
print("Loading custom vocabulary...")
vocab_dir = './vocab/'
token2idx, idx2token = load_vocabulary(vocab_dir)
latex_tokenizer = LaTeXTokenizer(token2idx, idx2token)
print(f"Vocabulary size: {latex_tokenizer.vocab_size}")



# Find best checkpoint
checkpoint_dir = "./checkpoint_latex_vocab/"
checkpoint_path = find_best_checkpoint(checkpoint_dir, exclude_latest=False)
#checkpoint_path=checkpoint_dir
print(f"\nLoading model from: {checkpoint_path}")

# Load image processor (from TrOCR)
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
image_processor = trocr_processor.image_processor

# Load model
model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
model.eval()

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Load test data
print("\nLoading test data...")
df2 = pd.read_table('./data/2014/caption.txt', header=None)
df2.rename(columns={0: "file_name", 1: "text"}, inplace=True)
df2['file_name'] = df2['file_name'].apply(lambda x: x+'.jpg')
df2 = df2.dropna()
df2.reset_index(drop=True, inplace=True)

print(f"Number of test examples: {len(df2)}")

# Run evaluation
print("\nStarting evaluation...\n")
results = evaluate_model(model, image_processor, latex_tokenizer, df2, './data/2014/', device)

# Print results
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Checkpoint used: {os.path.basename(checkpoint_path)}")
print(f"Character Error Rate (CER): {results['cer']:.4f}")
print(f"Word Error Rate (WER): {results['wer']:.4f}")
print(f"Exact Match Accuracy: {results['accuracy']:.4f}")
print(f"Exact Matches: {results['exact_matches']}/{results['total_samples']}")
print("="*50)