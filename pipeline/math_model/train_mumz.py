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
    
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      
                    dp[i][j-1],    
                    dp[i-1][j-1]  
                )
    
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
    r_len = len(reference)
    h_len = len(hypothesis)
    dp = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
    # Track operation types: 0=correct, 1=substitution, 2=deletion, 3=insertion
    backtrack = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
    for i in range(r_len + 1):
        dp[i][0] = i
        backtrack[i][0] = 2 # deletion
    for j in range(h_len + 1):
        dp[0][j] = j
        backtrack[0][j] = 3 # insertion

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                # Match - no operation needed
                dp[i][j] = dp[i-1][j-1]
                backtrack[i][j] = 0 # correct
            else:

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
        if torch.is_tensor(ref):
            ref = ref.cpu().numpy().tolist()
        if torch.is_tensor(hyp):
            hyp = hyp.cpu().numpy().tolist()
        
        # Remove special tokens
        ref_clean = [token for token in ref if token not in [pad_idx, start_idx, end_idx]]
        hyp_clean = [token for token in hyp if token not in [pad_idx, start_idx, end_idx]]
        metrics = compute_wer_detailed(ref_clean, hyp_clean)
        
        total_sub += metrics['substitutions']
        total_del += metrics['deletions']
        total_ins += metrics['insertions']
        total_cor += metrics['correct']
        total_words += metrics['total']
    
    # Calculate batch WER correctly: total errors / total words
    batch_wer = (total_sub + total_del + total_ins) / total_words if total_words > 0 else 0.0
    
    return {
        'wer': batch_wer, 
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


def make_5ch_from_image_gpu(img_path, blur_sigma=1.0, thick_radius=1, device="cuda"):
    """
    Converts image to 5 channels: [gray + 4 directional (0°, 45°, 90°, 135°)]
    Returns: torch.Tensor shape (5, H, W) - NO batch dimension
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unreadable: {img_path}")

    img = img.astype(np.float32) / 255.0

    img_t = torch.from_numpy(img).to(device)
    gray = img_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

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
    dirs = torch.cat(dirs, dim=1)  

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
        kernel_x = kernel.view(1, 1, -1, 1).repeat(dirs.shape[1], 1, 1, 1)
        kernel_y = kernel.view(1, 1, 1, -1).repeat(dirs.shape[1], 1, 1, 1)

        dirs = F.conv2d(dirs, kernel_x, padding=(radius, 0), groups=dirs.shape[1])
        dirs = F.conv2d(dirs, kernel_y, padding=(0, radius), groups=dirs.shape[1])

    # Normalize
    dirs = torch.sqrt(dirs / (dirs.amax(dim=(2, 3), keepdim=True) + 1e-12))

    # Stack grayscale + directional: (1, 1, H, W) + (1, 4, H, W) = (1, 5, H, W)
    five = torch.cat([gray, dirs], dim=1)
    
    return five.squeeze(0) # (1, 5, H, W) -> (5, H, W)



class MathExpressionDataset(Dataset):
    def __init__(self, csv_path, word2idx_path, base_image_dir, device="cuda", transform=None, subset_size=None):

        self.data_df = pd.read_csv(csv_path)
        if subset_size is not None:
            self.data_df = self.data_df.head(subset_size)
        
        with open(word2idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        
        self.base_image_dir = base_image_dir
        self.device = device
        self.transform = transform
        
        self.PAD_IDX = self.word2idx['<PAD>']
        self.START_IDX = self.word2idx['<START>']
        self.END_IDX = self.word2idx['<END>']
        self.UNK_IDX = self.word2idx.get('<UNK>', len(self.word2idx))

        unique_indices = len(set(self.word2idx.values()))

    def __len__(self):
        return len(self.data_df)

    def tokenize_latex(self, s):
        """
        Tokenize LaTeX string - MUST MATCH vocab_mumz.py tokenizer exactly!
        """
        tokens = []
        i = 0
        
        while i < len(s):
            # LaTeX commands (start with backslash)
            if s[i] == '\\':
                j = i + 1
                # Command names are alphabetic
                while j < len(s) and s[j].isalpha():
                    j += 1
                
                # If we found a command
                if j > i + 1:
                    tokens.append(s[i:j])  
                    i = j
                else:
                    # Special case: backslash followed by non-alpha
                    if j < len(s):
                        tokens.append(s[i:j+1])
                        i = j + 1
                    else:
                        tokens.append(s[i])
                        i += 1
        
            # Skip whitespace
            elif s[i].isspace():
                i += 1
            
            # Handle brackets, braces, and other special characters
            elif s[i] in '{}[]()^_=+-*/|<>!.,:;':
                tokens.append(s[i])
                i += 1
            
            # Single character token (digit, letter, punctuation)
            else:
                tokens.append(s[i])
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
                indices.append(self.UNK_IDX)
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
    
        try:
            fivech_tensor = make_5ch_from_image_gpu(
                image_path,
                blur_sigma=1.0,
                thick_radius=1,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process image {filename}: {e}")

        image_tensor = fivech_tensor.cpu()
        if self.transform:
            image_tensor = self.transform(image_tensor)
    
        label = row['label']
        # print(f"Label: {label}")
        tokens = self.tokenize_latex(label)
        # print(f"tokens:{tokens}")
        target = self.tokens_to_indices(tokens)
        # print(f"target: {target}")
        target_tensor = torch.tensor(target, dtype=torch.long)
        # print(f"target lengths in dataset:{len(target)}")
        return {
            'image': image_tensor,
            'target': target,
            'target_length': len(target),
            'label': label,
            'filename': filename
        }


def collate_fn(batch):
    """
    collate function to handle variable-length sequences AND variable image sizes
    """
    max_height = max(item['image'].shape[1] for item in batch)
    max_width = max(item['image'].shape[2] for item in batch)
    
    padded_images = []
    for item in batch:
        img = item['image']  # (5, H, W)
        C, H, W = img.shape
        
        # Create padded image with zeros
        padded = torch.zeros(C, max_height, max_width)
        padded[:, :H, :W] = img
        padded_images.append(padded)
    
    images = torch.stack(padded_images)  # (batch, 5, max_H, max_W)
    
    targets = [torch.tensor(item['target'], dtype=torch.long) for item in batch]
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    labels = [item['label'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'images': images,  # (batch, 5, max_H, max_W)
        'targets': targets_padded,
        'target_lengths': target_lengths,
        'labels': labels,
        'filenames': filenames
    }


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

        self.weight_noise_enabled = False
        self.weight_noise_sigma = 0.0
        self.weight_noise_cache = {}  # Store original weights
    
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate, betas=(0.9, 0.999), eps=epsilon, weight_decay=0.01
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6
        )

        self.criterion = nn.NLLLoss(ignore_index=self.pad_idx, reduction='mean')

        self.train_losses = []
        self.val_losses = []
        self.val_wers = []
        self.best_val_loss = float('inf')
        self.best_val_wer = float('inf')
        
    def enable_weight_noise(self, sigma=0.01):
        """
        Enable weight noise for training
        Args:
            sigma: Standard deviation of Gaussian noise (default: 0.01)
        """
        self.weight_noise_enabled = True
        self.weight_noise_sigma = sigma
        #print(f"Weight noise ENABLED with sigma={sigma}")
    
    def disable_weight_noise(self):
        """Disable weight noise"""
        self.weight_noise_enabled = False
        self.weight_noise_sigma = 0.0
        self.weight_noise_cache.clear()
        #print("Weight noise DISABLED")

    def apply_weight_noise(self):
        """
        Apply Gaussian noise to all model parameters
        Stores original weights for later restoration
        """
        if not self.weight_noise_enabled or self.weight_noise_sigma == 0:
            return
        
        self.weight_noise_cache.clear()
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.weight_noise_cache[f'encoder.{name}'] = param.data.clone()
                
                # Add Gaussian noise
                noise = torch.randn_like(param.data) * self.weight_noise_sigma
                param.data.add_(noise)
        
        # Apply noise to decoder parameters
        for name, param in self.decoder.named_parameters():
            if param.requires_grad:
                # Store original weight
                self.weight_noise_cache[f'decoder.{name}'] = param.data.clone()
                
                # Add Gaussian noise
                noise = torch.randn_like(param.data) * self.weight_noise_sigma
                param.data.add_(noise)    
                

    def restore_original_weights(self):
        """
        Restore original weights (remove noise)
        Call this after forward pass
        """
        if not self.weight_noise_cache:
            return
        
        for name, param in self.encoder.named_parameters():
            key = f'encoder.{name}'
            if key in self.weight_noise_cache:
                param.data.copy_(self.weight_noise_cache[key])
        
        for name, param in self.decoder.named_parameters():
            key = f'decoder.{name}'
            if key in self.weight_noise_cache:
                param.data.copy_(self.weight_noise_cache[key])
        
        self.weight_noise_cache.clear()

    def validate(self, val_loader, compute_wer_flag=True, use_beam_search=False, beam_width=5):
        """
        Validate the model
        Args:
            val_loader: Validation data loader
            compute_wer_flag: Whether to compute WER
            use_beam_search: If True, use beam search for decoding (no teacher forcing)
            beam_width: Beam width for beam search
        """
        self.encoder.eval()
        self.decoder.eval()
        val_loss = 0.0
        num_batches = len(val_loader)
        
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
                
                if use_beam_search:
                    #Use beam search for prediction (no teacher forcing)
                    predicted_sequences = []
                    for i in range(batch_size):
                        single_annotations = annotations[i:i+1]  # (1, L, D)
                        
                        # Beam search decode
                        pred_seq, _ = self.decoder.decode_beam_search(
                            annotations=single_annotations,
                            start_token=self.start_idx,
                            end_token=self.end_idx,
                            max_len=150,
                            beam_width=beam_width
                        )
                        predicted_sequences.append(pred_seq)
                    
                    # Compute loss using teacher forcing (for monitoring)
                    outputs, attentions = self.decoder(
                        annotations,
                        targets,
                        teacher_forcing_ratio=1.0
                    )
                    seq_len = outputs.size(1)
                    vocab_size = outputs.size(2)
                    outputs_flat = outputs.contiguous().view(-1, vocab_size)
                    targets_shifted = targets[:, 1:seq_len+1].contiguous().view(-1)
                    loss = self.criterion(outputs_flat, targets_shifted)
                    val_loss += loss.item()
                    
                    # Compute WER using beam search predictions
                    if compute_wer_flag:
                        references = []
                        hypotheses = []
                        for i in range(batch_size):
                            ref = targets[i, 1:].cpu().numpy().tolist()
                            if self.end_idx in ref:
                                end_pos = ref.index(self.end_idx)
                                ref = ref[:end_pos]
                            ref = [token for token in ref if token != self.pad_idx]
                            
                            hyp = predicted_sequences[i]
                            if hyp[0] == self.start_idx:
                                hyp = hyp[1:]
                            if len(hyp) > 0 and hyp[-1] == self.end_idx:
                                hyp = hyp[:-1]

                            hyp = [token for token in hyp if token != self.pad_idx]
                            
                            references.append(ref)
                            hypotheses.append(hyp)
                        
                        batch_metrics = batch_wer(references, hypotheses, 
                                                self.pad_idx, self.start_idx, self.end_idx)
                        
                        total_substitutions += batch_metrics['substitutions']
                        total_deletions += batch_metrics['deletions']
                        total_insertions += batch_metrics['insertions']
                        total_correct += batch_metrics['correct']
                        total_words += batch_metrics['total']
                
                else:
                    #Use teacher forcing for prediction 
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
                        predicted_indices = outputs.argmax(dim=2)
                    
                        references = []
                        hypotheses = []
                        for i in range(batch_size):
                            ref = targets[i, 1:].cpu().numpy().tolist()
                            if self.end_idx in ref:
                                end_pos = ref.index(self.end_idx)
                                ref = ref[:end_pos]
                            
                            hyp = predicted_indices[i].cpu().numpy().tolist()
                            if self.end_idx in hyp:
                                end_pos = hyp.index(self.end_idx)
                                hyp = hyp[:end_pos]
                            
                            ref = [token for token in ref if token != self.pad_idx]
                            hyp = [token for token in hyp if token != self.pad_idx]
                            
                            references.append(ref)
                            hypotheses.append(hyp)
                        
                        batch_metrics = batch_wer(references, hypotheses, 
                                                self.pad_idx, self.start_idx, self.end_idx)
                        
                        total_substitutions += batch_metrics['substitutions']
                        total_deletions += batch_metrics['deletions']
                        total_insertions += batch_metrics['insertions']
                        total_correct += batch_metrics['correct']
                        total_words += batch_metrics['total']
                
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{val_loss / (batch_idx + 1):.4f}'
                }
                if compute_wer_flag and total_words > 0:
                    current_wer = (total_substitutions + total_deletions + total_insertions) / total_words
                    postfix['avg_wer'] = f'{current_wer:.4f}'
                    if use_beam_search:
                        postfix['mode'] = 'beam_search'
                    else:
                        postfix['mode'] = 'teacher_forcing'
                pbar.set_postfix(postfix)
        
        avg_val_loss = val_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
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
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)

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
        Full training loop (WITHOUT beam search during validation)
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
            
            with open('/kaggle/input/crohme2014-vocab/ProccessedCrome2014Data/idx2word.pkl', 'rb') as f:
                idx2word = pickle.load(f)
            
            train_loss = self.train_epoch_minimal_debug(
                train_loader, epoch, 
                teacher_forcing_ratio=teacher_forcing_ratio,
                idx2word=idx2word
            )

            compute_wer_flag = (epoch % compute_wer_every == 0)
            val_loss, val_wer = self.validate(
                val_loader, 
                compute_wer_flag=compute_wer_flag,
                use_beam_search=False, 
                beam_width=5
            )
            
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start_time
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            if val_wer is not None and val_wer < self.best_val_wer:
                self.best_val_wer = val_wer
            
            self.save_checkpoint(epoch, val_loss, val_wer, is_best=is_best)
            
            print(f"\nEpoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_wer is not None:
                print(f"Val WER (Teacher Forcing): {val_wer:.4f}")
        
        print(f"\nTraining completed!")
    

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
            
            #Apply weight noise before forward pass
            self.apply_weight_noise()
            
            encoder_output = self.encoder(images)
            annotations = reshape_fcn_output(encoder_output)
            outputs, attentions = self.decoder(
                annotations,
                targets,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            seq_len = outputs.size(1)
            vocab_size = outputs.size(2)
            outputs_flat = outputs.contiguous().view(-1, vocab_size)
            targets_shifted = targets[:, 1:seq_len+1].contiguous().view(-1)
        
            # Compute loss
            loss = self.criterion(outputs_flat, targets_shifted)
            
            #Restore original weights before backward pass
            self.restore_original_weights()
            
            # Get probabilities and predictions
            import torch.nn.functional as F
            log_probs = F.log_softmax(outputs, dim=2)
            probs = torch.exp(log_probs)
            predicted_indices = outputs.argmax(dim=2)  # (batch, seq_len)
            
            for i in range(batch_size):
                ref = targets[i, 1:].cpu().numpy().tolist()
                if self.end_idx in ref:
                    end_pos = ref.index(self.end_idx)
                    ref = ref[:end_pos]
                ref = [t for t in ref if t != self.pad_idx]
                
                hyp = predicted_indices[i].cpu().numpy().tolist()
                if self.end_idx in hyp:
                    end_pos = hyp.index(self.end_idx)
                    hyp = hyp[:end_pos]
                hyp = [t for t in hyp if t != self.pad_idx]
            
                wer = compute_wer_basic(ref, hyp)
                
                if(batch_idx%100==0):
                    noise_status = f"Noise={self.weight_noise_sigma}" if self.weight_noise_enabled else "NoNoise"
                    print(f"Epoch={epoch} Batch={batch_idx}| "
                          f"Loss={loss.item():.4f} WER={wer:.4f} | {noise_status}")
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        return avg_epoch_loss
    
    def anneal_with_weight_noise(
        self,
        train_loader,
        val_loader,
        num_epochs=5,
        weight_noise_sigma=0.01,
        learning_rate=1e-4,
        teacher_forcing_ratio=0.5,
        compute_wer_every=1
    ):
        """
        Annealing phase: Fine-tune best model with weight noise
        This is the final training phase mentioned in WAP paper
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of annealing epochs (default: 5)
            weight_noise_sigma: Noise standard deviation (default: 0.01)
            learning_rate: Reduced learning rate for fine-tuning (default: 1e-4)
            teacher_forcing_ratio: Lower ratio for annealing (default: 0.5)
            compute_wer_every: Compute WER every N epochs
        """
        print("\n" + "="*80)
        print("STARTING ANNEALING PHASE WITH WEIGHT NOISE")
        print("="*80)
        print(f"Weight Noise Sigma: {weight_noise_sigma}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Teacher Forcing Ratio: {teacher_forcing_ratio}")
        print(f"Annealing Epochs: {num_epochs}")
        print("="*80 + "\n")
        
        self.enable_weight_noise(sigma=weight_noise_sigma)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Load idx2word for visualization
        with open('/kaggle/input/crohme2014-vocab/ProccessedCrome2014Data/idx2word.pkl', 'rb') as f:
            idx2word = pickle.load(f)
        
        best_anneal_wer = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"ANNEALING EPOCH {epoch}/{num_epochs}")
            print(f"{'='*80}")
            
            train_loss = self.train_epoch_minimal_debug(
                train_loader, 
                epoch, 
                teacher_forcing_ratio=teacher_forcing_ratio,
                idx2word=idx2word
            )
            
            # Validate WITHOUT weight noise (disable for evaluation)
            self.disable_weight_noise()
            
            compute_wer_flag = (epoch % compute_wer_every == 0)
            val_loss, val_wer = self.validate(
                val_loader, 
                compute_wer_flag=compute_wer_flag,
                use_beam_search=False,
                beam_width=5
            )
            
            # Re-enable weight noise for next epoch
            self.enable_weight_noise(sigma=weight_noise_sigma)
            
            epoch_time = time.time() - epoch_start_time
            
            is_best = False
            if val_wer is not None and val_wer < best_anneal_wer:
                best_anneal_wer = val_wer
                is_best = True
            
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
                'annealed': True,
                'weight_noise_sigma': weight_noise_sigma
            }
            
            anneal_path = os.path.join(self.checkpoint_dir, f'checkpoint_annealed_epoch_{epoch}.pth')
            torch.save(checkpoint, anneal_path)
            
            if is_best:
                best_anneal_path = os.path.join(self.checkpoint_dir, 'checkpoint_annealed_best.pth')
                torch.save(checkpoint, best_anneal_path)
                print(f"New best annealed model saved! WER: {val_wer:.4f}")
            
            print(f"\nAnnealing Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_wer is not None:
                print(f"Val WER: {val_wer:.4f} | Best Annealed WER: {best_anneal_wer:.4f}")
        
        self.disable_weight_noise()
        
        print("\n" + "="*80)
        print("ANNEALING PHASE COMPLETE!")
        print(f"Best Annealed WER: {best_anneal_wer:.4f}")
        print("="*80 + "\n")
    
    def evaluate_test_set(self, test_loader, beam_width=5, save_results=True):
        """
        Evaluate model on test set using beam search
        Args:
            test_loader: Test data loader
            beam_width: Beam width for beam search
            save_results: Whether to save detailed results to file
        Returns:
            Dictionary with test metrics and sample predictions
        """
        self.encoder.eval()
        self.decoder.eval()
        
        with open('/kaggle/input/crohme2014-vocab/ProccessedCrome2014Data/idx2word.pkl', 'rb') as f:
            idx2word = pickle.load(f)
        
        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_correct = 0
        total_words = 0
        
        results = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating Test Set (Beam Search)')
            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                labels = batch['labels']
                filenames = batch['filenames']
                batch_size = images.size(0)
            
                encoder_output = self.encoder(images)
                annotations = reshape_fcn_output(encoder_output)
                
                for i in range(batch_size):
                    single_annotations = annotations[i:i+1]  # (1, L, D)
                    
                    pred_seq, beam_score = self.decoder.decode_beam_search(
                        annotations=single_annotations,
                        start_token=self.start_idx,
                        end_token=self.end_idx,
                        max_len=150,
                        beam_width=beam_width
                    )
                
                    if isinstance(beam_score, (list, tuple)):
                        beam_score = beam_score[0] if len(beam_score) > 0 else 0.0
                    if torch.is_tensor(beam_score):
                        if beam_score.numel() == 1:
                            beam_score = beam_score.item()
                        else:
                            beam_score = beam_score[0].item()  
                    if isinstance(beam_score, np.ndarray):
                        if beam_score.size == 1:
                            beam_score = beam_score.item()
                        else:
                            beam_score = float(beam_score.flat[0])  
                    beam_score = float(beam_score)  
                
                    ref = targets[i, 1:].cpu().numpy().tolist()
                    if self.end_idx in ref:
                        end_pos = ref.index(self.end_idx)
                        ref = ref[:end_pos]
                    ref = [token for token in ref if token != self.pad_idx]
                    
                    hyp = pred_seq
                    if len(hyp) > 0 and hyp[0] == self.start_idx:
                        hyp = hyp[1:]
                    if len(hyp) > 0 and hyp[-1] == self.end_idx:
                        hyp = hyp[:-1]
                    hyp = [token for token in hyp if token != self.pad_idx]
                    
                    sample_metrics = compute_wer_detailed(ref, hyp)
                    
                    total_substitutions += sample_metrics['substitutions']
                    total_deletions += sample_metrics['deletions']
                    total_insertions += sample_metrics['insertions']
                    total_correct += sample_metrics['correct']
                    total_words += sample_metrics['total']
                
                    ref_symbols = [idx2word.get(int(t), f'<UNK:{t}>') for t in ref]
                    hyp_symbols = [idx2word.get(int(t), f'<UNK:{t}>') for t in hyp]
                    
                    result = {
                        'filename': filenames[i],
                        'ground_truth': labels[i],
                        'ground_truth_indices': ref,
                        'predicted_indices': hyp,
                        'ground_truth_symbols': ' '.join(ref_symbols),
                        'predicted_symbols': ' '.join(hyp_symbols),
                        'wer': sample_metrics['wer'],
                        'substitutions': sample_metrics['substitutions'],
                        'deletions': sample_metrics['deletions'],
                        'insertions': sample_metrics['insertions'],
                        'beam_score': beam_score  
                    }
                    results.append(result)
                
                if total_words > 0:
                    current_wer = (total_substitutions + total_deletions + total_insertions) / total_words
                    pbar.set_postfix({
                        'avg_wer': f'{current_wer:.4f}',
                        'samples': len(results)
                    })

        test_wer = (total_substitutions + total_deletions + total_insertions) / total_words if total_words > 0 else 0.0
        
        test_metrics = {
            'wer': test_wer,
            'substitutions': total_substitutions,
            'deletions': total_deletions,
            'insertions': total_insertions,
            'correct': total_correct,
            'total_words': total_words,
            'num_samples': len(results)
        }

        print("\n" + "="*80)
        print("TEST SET EVALUATION RESULTS (BEAM SEARCH)")
        print("="*80)
        print(f"Total Samples: {test_metrics['num_samples']}")
        print(f"Total Words: {test_metrics['total_words']}")
        print(f"Correct: {test_metrics['correct']}")
        print(f"Substitutions: {test_metrics['substitutions']}")
        print(f"Deletions: {test_metrics['deletions']}")
        print(f"Insertions: {test_metrics['insertions']}")
        print(f"Word Error Rate (WER): {test_metrics['wer']:.4f}")
        print("="*80)

        if save_results:
            results_file = os.path.join(self.log_dir, 'test_results_beam_search.txt')
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("TEST SET EVALUATION RESULTS (BEAM SEARCH)\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total Samples: {test_metrics['num_samples']}\n")
                f.write(f"Total Words: {test_metrics['total_words']}\n")
                f.write(f"Correct: {test_metrics['correct']}\n")
                f.write(f"Substitutions: {test_metrics['substitutions']}\n")
                f.write(f"Deletions: {test_metrics['deletions']}\n")
                f.write(f"Insertions: {test_metrics['insertions']}\n")
                f.write(f"Word Error Rate (WER): {test_metrics['wer']:.4f}\n")
                f.write("="*80 + "\n\n")
        
                f.write("DETAILED SAMPLE RESULTS:\n")
                f.write("="*80 + "\n\n")
                for idx, result in enumerate(results, 1):
                    f.write(f"Sample {idx}: {result['filename']}\n")
                    f.write(f"Ground Truth: {result['ground_truth']}\n")
                    f.write(f"GT Symbols:   {result['ground_truth_symbols']}\n")
                    f.write(f"Predicted:    {result['predicted_symbols']}\n")
                    f.write(f"WER: {result['wer']:.4f} | Sub: {result['substitutions']} | "
                        f"Del: {result['deletions']} | Ins: {result['insertions']} | "
                        f"Beam Score: {result['beam_score']:.4f}\n")
                    f.write("-"*80 + "\n\n")
            
            print(f"\nDetailed results saved to: {results_file}")
        
        return test_metrics, results


def teacher_forcing_schedule_linear(epoch):
    """Linear decay of teacher forcing ratio"""
    if epoch <= 50:
        return 1.0 - 0.5 * (epoch / 50)
    else:
        return 0.5
    
def teacher_forcing_schedule_constant(epoch):
    """Constant teacher forcing ratio of 0.8"""
    return 0.8

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


    
def main():
    """Main training script"""
    DATA_DIR = "/kaggle/input/crohme2014-vocab/ProccessedCrome2014Data"
    TRAIN_CSV = os.path.join(DATA_DIR, 'train_database_cleaned.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'test_database_cleaned.csv')
    WORD2IDX_PATH = "/kaggle/input/crohme2014-vocab/ProccessedCrome2014Data/word2idx.pkl"
    IDX2WORD_PATH = "/kaggle/input/crohme2014-vocab/ProccessedCrome2014Data/idx2word.pkl"
    BASE_IMAGE_DIR = DATA_DIR
    
    train_image_dir = os.path.join(BASE_IMAGE_DIR, 'train')
    test_image_dir = os.path.join(BASE_IMAGE_DIR, 'test')
    

    with open(WORD2IDX_PATH, 'rb') as f:
        word2idx = pickle.load(f)
    VOCAB_SIZE = len(set(word2idx.values()))
    PAD_IDX = word2idx['<PAD>']
    START_IDX = word2idx['<START>']
    END_IDX = word2idx['<END>']
    UNK_IDX = word2idx['<UNK>']
    
    EMBEDDING_DIM = 256
    DECODER_DIM = 256
    ENCODER_DIM = 128
    ATTENTION_DIM = 512
    COVERAGE_KERNEL_SIZE = 11
    
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    RHO = 0.95
    EPSILON = 1e-8
    NUM_WORKERS = 16
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    
    # 45-epoch checkpoint (WER = 0.85) trained without weight decay
    BEST_CHECKPOINT = '/kaggle/input/best-after-45-epochs/checkpoint_epoch_45.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = MathExpressionDataset(
        csv_path=TRAIN_CSV,
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=train_image_dir,
        device=device,
        transform=None,
    )
    
    val_dataset = MathExpressionDataset(
        csv_path=TEST_CSV,
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=test_image_dir,
        device=device,
        transform=None,
    )
    
    test_dataset = MathExpressionDataset(
        csv_path=TEST_CSV,
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=test_image_dir,
        device=device,
        transform=None,
        subset_size=None
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    encoder = FullyConvolutionalNetwork()
    decoder = GRUDecoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        decoder_dim=DECODER_DIM,
        encoder_dim=ENCODER_DIM,
        attention_dim=ATTENTION_DIM,
        kernel_size=COVERAGE_KERNEL_SIZE
    )
    
    print_model_summary(encoder, decoder)
    
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
    print("\n" + "="*80)
    print("LOADING BEST CHECKPOINT FROM 45 EPOCHS")
    print("="*80)
    print(f"Baseline Test WER after 45 epochs: 0.85")
    print(f"Goal: Reduce WER through weight noise annealing")
    print("="*80 + "\n")
    
    if os.path.exists(BEST_CHECKPOINT):
        print(f"Loading checkpoint: {BEST_CHECKPOINT}")
        trainer.load_checkpoint(BEST_CHECKPOINT)
        print("✅ Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {BEST_CHECKPOINT}")
    
    print("\n" + "="*80)
    print("STARTING ANNEALING PHASE WITH WEIGHT NOISE")
    print("="*80 + "\n")
    
    trainer.anneal_with_weight_noise(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,                  
        weight_noise_sigma=0.01,         
        learning_rate=1e-4,             
        teacher_forcing_ratio=0.6,      
        compute_wer_every=1              
    )
    
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET WITH BEAM SEARCH")
    print("="*80 + "\n")
    
    best_annealed_checkpoint = os.path.join(CHECKPOINT_DIR, 'checkpoint_annealed_best.pth')
    if os.path.exists(best_annealed_checkpoint):
        print(f"Loading best annealed checkpoint: {best_annealed_checkpoint}")
        trainer.load_checkpoint(best_annealed_checkpoint)
    else:
        print("Warning: Best annealed checkpoint not found, using current model state")
    
    test_metrics, test_results = trainer.evaluate_test_set(
        test_loader=test_loader,
        beam_width=5,  
        save_results=True
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL COMPARISON")
    print("="*80)
    print(f"Baseline WER (after 45 epochs):        0.8500")
    print(f"Final WER (after annealing):            {test_metrics['wer']:.4f}")
    
    improvement = 0.85 - test_metrics['wer']
    improvement_pct = (improvement / 0.85) * 100
    
    if improvement > 0:
        print(f"Improvement:                            {improvement:.4f} ({improvement_pct:.2f}%)")
        print("Model improved with weight noise annealing!")
    else:
        print(f"Change:                                 {improvement:.4f} ({improvement_pct:.2f}%)")
        print("No improvement - consider trying different hyperparameters")
    
    print("="*80)
    print("\nAnnealing Hyperparameters Used:")
    print("-" * 40)
    print(f"Epochs:                 10")
    print(f"Weight Noise Sigma:     0.01")
    print(f"Learning Rate:          1e-4")
    print(f"Teacher Forcing:        0.6 (reduced from 0.8)")
    print(f"Beam Width:             5")
    print("="*80)

if __name__ == '__main__':
    main()