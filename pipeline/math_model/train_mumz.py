# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pickle
import os
import time
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# # Import your model classes
from model_mumz import FullyConvolutionalNetwork, GRUDecoder, reshape_fcn_output
import numpy as np
from collections import defaultdict
import cv2

# ============================================================================
# WER CALCULATION FUNCTION
# ============================================================================

def compute_wer_basic(reference, hypothesis):
    """
    Compute Word Error Rate (WER) using Levenshtein distance
    """
    r_len = len(reference)
    h_len = len(hypothesis)
    
    # Create DP table
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    
    # Initialize
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    # Calculate WER
    total_tokens = r_len
    if total_tokens == 0:
        return 0.0 if h_len == 0 else float('inf')
    
    wer = dp[r_len][h_len] / total_tokens
    return wer


def compute_wer_detailed(reference, hypothesis):
    """
    Compute Word Error Rate (WER) using Levenshtein distance
    Args:
    reference: List of reference tokens (ground truth)
    hypothesis: List of predicted tokens
    Returns:
    Dictionary with WER metrics:
    - wer: Word Error Rate (float)
    - substitutions: Number of substitutions
    - deletions: Number of deletions
    - insertions: Number of insertions
    - correct: Number of correct tokens
    - total: Total number of tokens in reference
    """
    # Initialize DP table for edit distance
    r_len = len(reference)
    h_len = len(hypothesis)
    # Create (r_len+1) x (h_len+1) matrix
    dp = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
    # Track operation types: 0=correct, 1=substitution, 2=deletion, 3=insertion
    backtrack = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
    # Initialize first row and column
    for i in range(r_len + 1):
        dp[i][0] = i
        backtrack[i][0] = 2 # deletion
    for j in range(h_len + 1):
        dp[0][j] = j
        backtrack[0][j] = 3 # insertion
    # Fill DP table
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                # Match - no operation needed
                dp[i][j] = dp[i-1][j-1]
                backtrack[i][j] = 0 # correct
            else:
                # Find minimum cost operation
                substitution = dp[i-1][j-1] + 1
                deletion = dp[i-1][j] + 1
                insertion = dp[i][j-1] + 1
                min_cost = min(substitution, deletion, insertion)
                dp[i][j] = min_cost
                if min_cost == substitution:
                    backtrack[i][j] = 1 # substitution
                elif min_cost == deletion:
                    backtrack[i][j] = 2 # deletion
                else:
                    backtrack[i][j] = 3 # insertion
    # Backtrack to count operations
    i, j = r_len, h_len
    n_sub = 0
    n_del = 0
    n_ins = 0
    n_cor = 0
    while i > 0 or j > 0:
        operation = backtrack[i][j]
        if operation == 0: # correct
            n_cor += 1
            i -= 1
            j -= 1
        elif operation == 1: # substitution
            n_sub += 1
            i -= 1
            j -= 1
        elif operation == 2: # deletion
            n_del += 1
            i -= 1
        elif operation == 3: # insertion
            n_ins += 1
            j -= 1
    # Calculate WER
    # WER = (NW_sub + NW_del + NW_ins) / NW
    # where NW = total words in reference
    total_words = r_len
    if total_words == 0:
        wer = 0.0 if h_len == 0 else float('inf')
    else:
        wer = (n_sub + n_del + n_ins) / total_words
    return {
        'wer': wer,
        'substitutions': n_sub,
        'deletions': n_del,
        'insertions': n_ins,
        'correct': n_cor,
        'total': total_words
    }

def batch_wer(references, hypotheses, pad_idx=0, start_idx=1, end_idx=2):
    """
    Compute average WER for a batch of sequences
    """
    batch_size = len(references)
    total_sub = 0
    total_del = 0
    total_ins = 0
    total_cor = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        # Convert tensors to lists if needed
        if torch.is_tensor(ref):
            ref = ref.cpu().numpy().tolist()
        if torch.is_tensor(hyp):
            hyp = hyp.cpu().numpy().tolist()
        
        # Remove special tokens
        ref_clean = [token for token in ref if token not in [pad_idx, start_idx, end_idx]]
        hyp_clean = [token for token in hyp if token not in [pad_idx, start_idx, end_idx]]
        
        # Compute WER for this pair
        metrics = compute_wer_detailed(ref_clean, hyp_clean)
        
        total_sub += metrics['substitutions']
        total_del += metrics['deletions']
        total_ins += metrics['insertions']
        total_cor += metrics['correct']
        total_words += metrics['total']
    
    # Calculate batch WER correctly: total errors / total words
    batch_wer = (total_sub + total_del + total_ins) / total_words if total_words > 0 else 0.0
    
    return {
        'wer': batch_wer,  # ← Correct: total errors / total words
        'substitutions': total_sub,
        'deletions': total_del,
        'insertions': total_ins,
        'correct': total_cor,
        'total': total_words
    }

def make_grayscale_from_image_gpu(img_path, out_size=None, device="cuda"):
    """
    Converts image to 1-channel grayscale.
    Returns: torch.Tensor shape (1, 1, H, W)
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")
    img = img.astype(np.float32) / 255.0
    if out_size is not None:
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_LINEAR)
    H, W = img.shape
    img_t = torch.from_numpy(img).to(device)
    gray = img_t.unsqueeze(0).unsqueeze(0)
    return gray


def make_5ch_from_image_gpu(img_path, out_size=None, blur_sigma=1.0, thick_radius=1, device="cuda"):
    """
    Converts image to 5 channels: [gray + 4 directional (0°, 45°, 90°, 135°)]
    Returns: torch.Tensor shape (1, 5, H, W)
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

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
    ori = torch.atan2(gy, gx)  # radians in [-π, π]

    # Directional bins: 4 directions (0°, 45°, 90°, 135°)
    nbins = 4
    bin_edges = torch.linspace(-np.pi, np.pi, nbins + 1, device=device)
    dirs = []
    for b in range(nbins):
        mask = ((ori >= bin_edges[b]) & (ori < bin_edges[b + 1])).float()
        dirs.append(mag * mask)
    dirs = torch.cat(dirs, dim=0)  # shape: (4, 1, H, W)

    # Optional thickening (dilation-like effect)
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

        dirs = dirs.permute(1, 0, 2, 3)
        dirs = F.conv2d(dirs, kernel_x, padding=(radius, 0), groups=dirs.shape[1])
        dirs = F.conv2d(dirs, kernel_y, padding=(0, radius), groups=dirs.shape[1])
        dirs = dirs.permute(1, 0, 2, 3)

    # Normalize
    dirs = torch.sqrt(dirs / (dirs.amax(dim=(2, 3), keepdim=True) + 1e-12))

    # Stack grayscale + directional
    five = torch.cat([gray, dirs], dim=0)
    return five.unsqueeze(0)  # (1, 5, H, W)

# ============================================================================
# DATASET CLASS - COMPUTES GRAYSCALE ON-THE-FLY
# ============================================================================
class MathExpressionDataset(Dataset):
    def __init__(self, csv_path, word2idx_path, base_image_dir, device="cuda", transform=None, subset_size=None):
        # Load CSV database
        self.data_df = pd.read_csv(csv_path)

        # ✅ Limit to first `subset_size` rows if specified
        if subset_size is not None:
            self.data_df = self.data_df.head(subset_size)
        # Load vocabulary mapping
        with open(word2idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        self.base_image_dir = base_image_dir
        self.device = device
        self.transform = transform
        # Special tokens
        self.PAD_IDX = self.word2idx['<PAD>']  # Assuming keys exist; adjust if needed
        self.START_IDX = self.word2idx['<START>']
        self.END_IDX = self.word2idx['<END>']
        self.UNK_IDX = self.word2idx['<UNK>']
        # Math word mapping
        self.MATH_WORD_MAP = {
            "sin": "\\sin", "cos": "\\cos", "tan": "\\tan", "cot": "\\cot",
            "sec": "\\sec", "csc": "\\csc",
            "sinh": "\\sinh", "cosh": "\\cosh", "tanh": "\\tanh", "coth": "\\coth",
            "log": "\\log", "ln": "\\ln", "exp": "\\exp",
            "lim": "\\lim", "max": "\\max", "min": "\\min", "sup": "\\sup", "inf": "\\inf",
            "det": "\\det", "rank": "\\rank", "trace": "\\trace",
            "arg": "\\arg", "deg": "\\deg", "mod": "\\mod", "ker": "\\ker",
            "gcd": "\\gcd", "pr": "\\Pr", "hom": "\\hom",
            "arcsin": "\\arcsin", "arccos": "\\arccos", "arctan": "\\arctan",
            "arcsec": "\\arcsec", "arccot": "\\arccot", "arccsc": "\\arccsc"
        }
        unique_indices = len(set(self.word2idx.values()))

    def __len__(self):
        return len(self.data_df)

    def tokenize_latex(self, s):
        """
        Tokenize LaTeX string using the same logic as your vocab builder
        Args:
        s: LaTeX string
        Returns:
        List of token strings
        """
        tokens = []
        i = 0
        L = len(s)
        while i < L:
            ch = s[i]
            # 1) LaTeX commands starting with backslash
            if ch == '\\':
                j = i + 1
                while j < L and s[j].isalpha():
                    j += 1
                cmd = s[i:j]
                tokens.append(cmd)
                i = j
                continue
            # 2) Underscore (_) - keep as single token
            if ch == '_':
                tokens.append(ch)
                i += 1
                continue
            # 3) Numbers (with decimal points) - split into individual digits/dots
            if ch.isdigit():
                j = i
                while j < L and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                num = s[i:j]
                tokens.extend(list(num)) # Split into characters
                i = j
                continue
            # 4) Letters: detect contiguous letters - check if it's a known math word
            if ch.isalpha():
                j = i
                while j < L and s[j].isalpha():
                    j += 1
                word = s[i:j]
                low = word.lower()
                # If word is a known math word, map to LaTeX version
                if word in self.MATH_WORD_MAP:
                    tokens.append(self.MATH_WORD_MAP[word])
                elif low in self.MATH_WORD_MAP:
                    tokens.append(self.MATH_WORD_MAP[low])
                else:
                    # Split into individual letters
                    tokens.extend(list(word))
                i = j
                continue
            # 5) Whitespace -> skip
            if ch.isspace():
                i += 1
                continue
            # 6) Other single-character tokens (operators, punctuation, braces, etc.)
            tokens.append(ch)
            i += 1
        return tokens

    def tokens_to_indices(self, tokens):
        """
        Convert token strings to indices using vocabulary
        Args:
        tokens: List of token strings
        Returns:
        List of token indices
        """
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                # Token not in vocab - use UNK
                indices.append(self.UNK_IDX)
        # Add START and END tokens
        token_sequence = [self.START_IDX] + indices + [self.END_IDX]
        return token_sequence

    def __getitem__(self, idx):
        """
        Get a single sample
        Converts grayscale image → 5-channel tensor using make_5ch_from_image_gpu()
        """
        row = self.data_df.iloc[idx]
        filename = row['filename']
        image_path = os.path.join(self.base_image_dir, filename)
    
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
    
        # ✅ Convert grayscale image to 5-channel tensor on the fly
        try:
            fivech_tensor = make_5ch_from_image_gpu(
                image_path,
                out_size=(800, 240),   # (W, H)
                blur_sigma=1.0,
                thick_radius=1,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process image {filename}: {e}")
    
        # Reshape to (5, 240, 800)
        image_tensor = fivech_tensor.reshape(5, 240, 800)
    
        # Ensure correct shape
        if image_tensor.shape != torch.Size([5, 240, 800]):
            raise ValueError(
                f"Unexpected tensor shape! Expected [5, 240, 800], got {image_tensor.shape}"
            )
    
        # Move to CPU to save GPU memory
        image_tensor = image_tensor.cpu()
    
        # Optional transforms
        if self.transform:
            image_tensor = self.transform(image_tensor)
    
        # Tokenize and encode label
        label = row['normalized_label']
        #print(f"Label: {label}")
        tokens = self.tokenize_latex(label)
        #print(f"tokens:{tokens}")
        target = self.tokens_to_indices(tokens)
        #print(f"target: {target}")
        target_tensor = torch.tensor(target, dtype=torch.long)
        #print(f"target lengths in dataset:{len(target)}")
        return {
            'image': image_tensor,
            'target': target,
            'target_length': len(target),
            'label': label,
            'filename': filename
        }



def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Args:
    batch: List of samples from dataset
    Returns:
    Dictionary with batched and padded data
    """
    # Extract images (all same size: 1 x 240 x 800)
    images = torch.stack([item['image'] for item in batch])  # (batch, 5, 240, 800)
    # Extract targets (variable length)
    targets = [torch.tensor(item['target'], dtype=torch.long) for item in batch]
    # Get original lengths before padding
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    # Pad sequences to same length in batch
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0) # 0 is PAD
    # Extract other info
    labels = [item['label'] for item in batch]
    filenames = [item['filename'] for item in batch]
    return {
        'images': images, # (batch, 1, 240, 800)
        'targets': targets_padded, # (batch, max_seq_len)
        'target_lengths': target_lengths, # (batch,)
        'labels': labels, # List of strings
        'filenames': filenames # List of strings
    }

# ============================================================================
# TRAINER CLASS (Same as before)
# ============================================================================
class MathExpressionTrainer:
    """
    Training class for the WAP (Watch, Attend, Parse) model
    """
    def __init__(
        self,
        encoder,
        decoder,
        device,
        pad_idx=0,
        start_idx=1,
        end_idx=2,
        learning_rate=1e-3,
        rho=0.95,
        epsilon=1e-8,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        """
        Args:
        encoder: FCN encoder model
        decoder: GRU decoder with attention
        device: torch.device (cuda or cpu)
        pad_idx: Index of padding token in vocabulary (default: 0)
        learning_rate: Learning rate for Adadelta (default: 1.0)
        rho: Decay rate for Adadelta (default: 0.95)
        epsilon: Term added to denominator for numerical stability (default: 1e-6)
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
        """
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        # Create checkpoint and log directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        # AdamW optimizer for both encoder and decoder

        # self.optimizer = torch.optim.Adam(
        #     list(self.encoder.parameters()) + list(self.decoder.parameters()),
        #     lr=learning_rate, betas=(0.9, 0.999), eps=epsilon, weight_decay=0.0
        # )
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate, betas=(0.9, 0.999), eps=epsilon, weight_decay=0.01
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6
        )

        # NLLLoss with padding token ignored
        self.criterion = nn.NLLLoss(ignore_index=self.pad_idx, reduction='mean')
        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.val_wers = []
        self.best_val_loss = float('inf')
        self.best_val_wer = float('inf')

    def validate(self, val_loader, compute_wer_flag=True):
        """Validate the model"""
        self.encoder.eval()
        self.decoder.eval()
        val_loss = 0.0
        num_batches = len(val_loader)
        
        # WER tracking - accumulate error counts and word counts
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_correct = 0
        total_words = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                target_lengths = batch['target_lengths']
                batch_size = images.size(0)
                
                # Forward pass through encoder
                encoder_output = self.encoder(images)
                annotations = reshape_fcn_output(encoder_output)
                
                # Forward pass through decoder
                outputs, attentions = self.decoder(
                    annotations,
                    targets,
                    teacher_forcing_ratio=1.0
                )
                
                # Compute loss
                seq_len = outputs.size(1)
                vocab_size = outputs.size(2)
                outputs_flat = outputs.contiguous().view(-1, vocab_size)
                targets_shifted = targets[:, 1:seq_len+1].contiguous().view(-1)
                loss = self.criterion(outputs_flat, targets_shifted)
                val_loss += loss.item()
                
                # Compute WER if requested
                if compute_wer_flag:
                    # Get predictions
                    predicted_indices = outputs.argmax(dim=2)
                    
                    # Prepare references and hypotheses
                    references = []
                    hypotheses = []
                    for i in range(batch_size):
                        # Reference: remove START token, keep until END or PAD
                        ref = targets[i, 1:].cpu().numpy().tolist()
                        if self.end_idx in ref:
                            end_pos = ref.index(self.end_idx)
                            ref = ref[:end_pos]
                        
                        # Hypothesis: predicted tokens
                        hyp = predicted_indices[i].cpu().numpy().tolist()
                        if self.end_idx in hyp:
                            end_pos = hyp.index(self.end_idx)
                            hyp = hyp[:end_pos]
                        
                        # Remove PAD tokens
                        ref = [token for token in ref if token != self.pad_idx]
                        hyp = [token for token in hyp if token != self.pad_idx]
                        
                        references.append(ref)
                        hypotheses.append(hyp)
                    
                    # Compute WER for this batch
                    batch_metrics = batch_wer(references, hypotheses, 
                                             self.pad_idx, self.start_idx, self.end_idx)
                    
                    # Accumulate counts (NOT WER values!)
                    total_substitutions += batch_metrics['substitutions']
                    total_deletions += batch_metrics['deletions']
                    total_insertions += batch_metrics['insertions']
                    total_correct += batch_metrics['correct']
                    total_words += batch_metrics['total']
                
                # Update progress bar
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{val_loss / (batch_idx + 1):.4f}'
                }
                if compute_wer_flag and total_words > 0:
                    # Calculate WER on the fly from accumulated counts
                    current_wer = (total_substitutions + total_deletions + total_insertions) / total_words
                    postfix['avg_wer'] = f'{current_wer:.4f}'
                pbar.set_postfix(postfix)
        
        avg_val_loss = val_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        # Calculate final WER from accumulated counts
        if compute_wer_flag and total_words > 0:
            avg_wer = (total_substitutions + total_deletions + total_insertions) / total_words
            self.val_wers.append(avg_wer)
        else:
            avg_wer = None
        
        return avg_val_loss, avg_wer

    
    def save_checkpoint(self, epoch, val_loss, val_wer=None, is_best=False):
        """
        Save model checkpoint
        Args:
        epoch: Current epoch number
        val_loss: Validation loss
        val_wer: Validation WER (optional)
        is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_wers': self.val_wers,
            'val_loss': val_loss,
            'val_wer': val_wer,
        }
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
        # Save periodic checkpoint
        epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_wers = checkpoint.get('val_wers', [])
        epoch = checkpoint['epoch']
        return epoch + 1


    def train(
        self,
        train_loader,
        val_loader,
        num_epochs,
        start_epoch=1,
        teacher_forcing_schedule=None,
        resume_from=None,
        compute_wer_every=1
    ):
        """
        Full training loop
        Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Total number of epochs to train
        start_epoch: Starting epoch (for resuming training)
        teacher_forcing_schedule: Function that takes epoch and returns teacher_forcing_ratio
        resume_from: Path to checkpoint to resume from
        compute_wer_every: Compute WER every N epochs (default: 1)
        """
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start_time = time.time()
            if teacher_forcing_schedule:
                teacher_forcing_ratio = teacher_forcing_schedule(epoch)
            else:
                teacher_forcing_ratio = 1.0
            # train_loss = self.train_epoch(train_loader, epoch, teacher_forcing_ratio=teacher_forcing_ratio)
            with open('/kaggle/input/vocab-1-nov/idx2word.pkl', 'rb') as f:
                    idx2word = pickle.load(f)
            train_loss = self.train_epoch_minimal_debug(train_loader, epoch, teacher_forcing_ratio=teacher_forcing_ratio,idx2word=idx2word )


            # Compute WER every N epochs
            compute_wer_flag = (epoch % compute_wer_every == 0)
            val_loss, val_wer = self.validate(val_loader, compute_wer_flag=compute_wer_flag)
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start_time
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            if val_wer is not None and val_wer < self.best_val_wer:
                self.best_val_wer = val_wer
            self.save_checkpoint(epoch, val_loss, val_wer, is_best=is_best)
        print(f"Training completed!")
    
    
    def train_epoch_minimal_debug(self, train_loader, epoch, teacher_forcing_ratio=1.0, idx2word=None):
        """
        Minimal debug training loop with vocab symbols
        idx2word: Dictionary mapping token indices to vocabulary symbols
        """
        self.encoder.train()
        self.decoder.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            labels = batch['labels']
            
            batch_size = images.size(0)
            
            # ====================================================================
            # FORWARD PASS
            # ====================================================================
            encoder_output = self.encoder(images)
            annotations = reshape_fcn_output(encoder_output)
            outputs, attentions = self.decoder(
                annotations,
                targets,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # Prepare for loss
            seq_len = outputs.size(1)
            vocab_size = outputs.size(2)
            outputs_flat = outputs.contiguous().view(-1, vocab_size)
            targets_shifted = targets[:, 1:seq_len+1].contiguous().view(-1)
        
            # Compute loss
            loss = self.criterion(outputs_flat, targets_shifted)
            
            # Get probabilities and predictions
            import torch.nn.functional as F
            log_probs = F.log_softmax(outputs, dim=2)
            probs = torch.exp(log_probs)
            predicted_indices = outputs.argmax(dim=2)  # (batch, seq_len)
            
            # ====================================================================
            # COMPUTE WER AND PRINT FOR EACH SAMPLE
            # ====================================================================
            for i in range(batch_size):
                # Get reference (ground truth)
                ref = targets[i, 1:].cpu().numpy().tolist()
                if self.end_idx in ref:
                    end_pos = ref.index(self.end_idx)
                    ref = ref[:end_pos]
                ref = [t for t in ref if t != self.pad_idx]
                
                # Get hypothesis (prediction)
                hyp = predicted_indices[i].cpu().numpy().tolist()
                if self.end_idx in hyp:
                    end_pos = hyp.index(self.end_idx)
                    hyp = hyp[:end_pos]
                hyp = [t for t in hyp if t != self.pad_idx]
            
                # Compute WER
                wer = compute_wer_basic(ref, hyp)
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Check if gradient clipping will be applied
                will_clip = False
                for param in list(self.encoder.parameters()) + list(self.decoder.parameters()):
                    if param.grad is not None:
                        if param.grad.norm().item() > 5.0:
                            will_clip = True
                            break
                
                # Convert indices to symbols using idx2word
                # if idx2word is not None:
                #     target_symbols = [idx2word.get(int(t), f'<UNK:{t}>') for t in ref]
                #     predicted_symbols = [idx2word.get(int(t), f'<UNK:{t}>') for t in hyp]
                #     target_str = ' '.join(target_symbols)
                #     predicted_str = ' '.join(predicted_symbols)
                # else:
                #     target_str = ' '.join(str(t) for t in ref)
                #     predicted_str = ' '.join(str(t) for t in hyp)
            
                # Print minimal info
                print(f"Epoch={epoch} Batch={batch_idx} Sample={i} | "
                      f"Loss={loss.item():.4f} WER={wer:.4f} | "
                   #f"Output seq len:{seq_len} | "
                      f"LR={current_lr:.2e} TeacherForcing={teacher_forcing_ratio:.2f} Clip={will_clip} | "
                      #f"EncIn={images.shape} EncOut={encoder_output.shape} "
                      #f"Reshaped={annotations.shape} DecOut={outputs.shape}"
                        )
                
                # print(f"Target Indices:  {ref}")
                # print(f"Target Symbols:  {target_str}")
                # print(f"Pred Indices:    {hyp}")
                # print(f"Pred Symbols:    {predicted_str}\n")
            
            # ====================================================================
            # BACKWARD PASS
            # ====================================================================
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        return avg_epoch_loss


def teacher_forcing_schedule_linear(epoch):
    """Linear decay of teacher forcing ratio"""
    if epoch <= 50:
        return 1.0 - 0.5 * (epoch / 50)
    else:
        return 0.5

# ============================================================================
# UTILITY FUNCTION FOR PARAMETER COUNTING
# ============================================================================
def count_parameters(model, trainable_only=True):
    """
    Count the number of parameters in a model
    Args:
    model: PyTorch model
    trainable_only: If True, count only trainable parameters
    Returns:
    Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def print_model_summary(encoder, decoder):
    """
    Print detailed parameter summary for encoder and decoder
    Args:
    encoder: Encoder model
    decoder: Decoder model
    """
    # Encoder parameters
    encoder_trainable = count_parameters(encoder, trainable_only=True)
    encoder_total = count_parameters(encoder, trainable_only=False)
    # Decoder parameters
    decoder_trainable = count_parameters(decoder, trainable_only=True)
    decoder_total = count_parameters(decoder, trainable_only=False)
    # Total parameters
    total_trainable = encoder_trainable + decoder_trainable
    total_all = encoder_total + decoder_total




# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================
def main():
    """Main training script"""
    DATA_DIR = '/kaggle/input/mathwritting-smaller-800x480-total-60k/smaller' # UPDATE THIS
    TRAIN_CSV = os.path.join(DATA_DIR, 'train_database.csv')
    VAL_CSV = os.path.join(DATA_DIR, 'val_database.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'test_database.csv')
    WORD2IDX_PATH = os.path.join('/kaggle/input/vocab-1-nov', 'word2idx.pkl')
    IDX2WORD_PATH = os.path.join('/kaggle/input/vocab-1-nov', 'idx2word.pkl')
    BASE_IMAGE_DIR = '/kaggle/input/mathwritting-smaller-800x480-total-60k/smaller' # UPDATE THIS PATH
    train_image_dir = os.path.join(BASE_IMAGE_DIR, 'train')
    val_image_dir = os.path.join(BASE_IMAGE_DIR, 'val')
    # ========================================================================
    # HYPERPARAMETERS
    # ========================================================================
    with open(WORD2IDX_PATH, 'rb') as f:
        word2idx = pickle.load(f)
    VOCAB_SIZE = len(set(word2idx.values())) # Unique token indices
    PAD_IDX = word2idx['<PAD>']  # Adjust keys as needed
    START_IDX = word2idx['<START>']
    END_IDX = word2idx['<END>']
    UNK_IDX = word2idx['<UNK>']
    # Model hyperparameters
    EMBEDDING_DIM = 256
    DECODER_DIM = 256
    ENCODER_DIM = 128
    ATTENTION_DIM = 512
    COVERAGE_KERNEL_SIZE = 11
    # Training hyperparameters
    BATCH_SIZE = 1
    NUM_EPOCHS = 6
    LEARNING_RATE = 1e-3
    RHO = 0.95
    EPSILON = 1e-8
    NUM_WORKERS = 16
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    RESUME_FROM = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # ========================================================================
    # CREATE DATASETS AND DATALOADERS
    # ========================================================================
    # Define base directories
    train_dataset = MathExpressionDataset(
        csv_path=TRAIN_CSV,
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=train_image_dir, # ADD THIS
        device=device, # ADD THIS
        transform=None,
        #subset_size=30000 
    )
    #Load idx2word
    with open(IDX2WORD_PATH, 'rb') as f:
        idx2word = pickle.load(f)
    
    val_dataset = MathExpressionDataset(
        csv_path=VAL_CSV,
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=val_image_dir, # ADD THIS
        device=device, # ADD THIS
        transform=None,
        #subset_size=3000
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    # ========================================================================
    # CREATE MODELS
    # ========================================================================
    encoder = FullyConvolutionalNetwork()
    decoder = GRUDecoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        decoder_dim=DECODER_DIM,
        encoder_dim=ENCODER_DIM,
        attention_dim=ATTENTION_DIM,
        kernel_size=COVERAGE_KERNEL_SIZE
    )
    # Print detailed parameter summary
    print_model_summary(encoder, decoder)
    # ========================================================================
    # CREATE TRAINER AND START TRAINING
    # ========================================================================
    trainer = MathExpressionTrainer(
        encoder=encoder,
        decoder=decoder,
        device=device,
        pad_idx=PAD_IDX,
        start_idx=START_IDX,
        end_idx=END_IDX,
        learning_rate=LEARNING_RATE,
        rho=RHO,
        epsilon=EPSILON,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        teacher_forcing_schedule=teacher_forcing_schedule_linear,
        resume_from=RESUME_FROM,
        compute_wer_every=1 # Compute WER every epoch
    )

# if __name__ == '__main__':
#     main()
