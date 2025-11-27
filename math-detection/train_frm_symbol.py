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
# from model_mumz import FullyConvolutionalNetwork, GRUDecoder, reshape_fcn_output
import numpy as np
from collections import defaultdict
import cv2
import csv

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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
    dirs = torch.cat(dirs, dim=1)  # ✅ Changed from dim=0 to dim=1: (1, 4, H, W)

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
    
    # ✅ Remove batch dimension: (1, 5, H, W) -> (5, H, W)
    return five.squeeze(0)

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
                    tokens.append(s[i:j])  # e.g., "\sin"
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
                blur_sigma=1.0,
                thick_radius=1,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process image {filename}: {e}")

        # Move to CPU to save GPU memory
        image_tensor = fivech_tensor.cpu()
    
        # Optional transforms
        if self.transform:
            image_tensor = self.transform(image_tensor)
    
        # Tokenize and encode label
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
    Custom collate function to handle variable-length sequences AND variable image sizes
    """
    # ✅ Find max height and width in batch
    max_height = max(item['image'].shape[1] for item in batch)
    max_width = max(item['image'].shape[2] for item in batch)
    
    # ✅ Pad images to same size in batch
    padded_images = []
    for item in batch:
        img = item['image']  # (5, H, W)
        C, H, W = img.shape
        
        # Create padded image with zeros
        padded = torch.zeros(C, max_height, max_width)
        padded[:, :H, :W] = img
        padded_images.append(padded)
    
    images = torch.stack(padded_images)  # (batch, 5, max_H, max_W)
    
    # ✅ FIXED: Convert lists to tensors
    targets = [torch.tensor(item['target'], dtype=torch.long) for item in batch]
    
    labels = [item['label'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'images': images,
        'targets': targets,  # ✅ List of tensors (not list of lists)
        'labels': labels,
        'filenames': filenames
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
        self.gradient_stats = {
            'epoch': [],
            'batch': [],
            'encoder_grad_norm': [],
            'decoder_grad_norm': [],
            'total_grad_norm': [],
            'encoder_grad_max': [],
            'decoder_grad_max': [],
            'encoder_grad_min': [],
            'decoder_grad_min': [],
            'loss': [],
            'learning_rate': []
        }

    def compute_gradient_norm(self, model, norm_type=2):
        """
        Compute gradient norm for a model
        Args:
            model: PyTorch model
            norm_type: Type of norm (default: 2 for L2 norm)
        Returns:
            total_norm: Total gradient norm
            max_grad: Maximum gradient value
            min_grad: Minimum gradient value
        """
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        
        if len(parameters) == 0:
            return 0.0, 0.0, 0.0
        
        device = parameters[0].grad.device
        
        # Compute total norm
        if norm_type == float('inf'):
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                norm_type
            )
        
        # Get max and min gradient values
        max_grad = max(p.grad.detach().abs().max().item() for p in parameters)
        min_grad = min(p.grad.detach().abs().min().item() for p in parameters)
        
        return total_norm.item(), max_grad, min_grad

    def log_gradients(self, epoch, batch_idx, loss):
        """
        Log gradient statistics
        """
        # Compute gradient norms
        encoder_norm, encoder_max, encoder_min = self.compute_gradient_norm(self.encoder)
        decoder_norm, decoder_max, decoder_min = self.compute_gradient_norm(self.decoder)
        
        # Total gradient norm
        total_norm = (encoder_norm ** 2 + decoder_norm ** 2) ** 0.5
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Store statistics
        self.gradient_stats['epoch'].append(epoch)
        self.gradient_stats['batch'].append(batch_idx)
        self.gradient_stats['encoder_grad_norm'].append(encoder_norm)
        self.gradient_stats['decoder_grad_norm'].append(decoder_norm)
        self.gradient_stats['total_grad_norm'].append(total_norm)
        self.gradient_stats['encoder_grad_max'].append(encoder_max)
        self.gradient_stats['decoder_grad_max'].append(decoder_max)
        self.gradient_stats['encoder_grad_min'].append(encoder_min)
        self.gradient_stats['decoder_grad_min'].append(decoder_min)
        self.gradient_stats['loss'].append(loss)
        self.gradient_stats['learning_rate'].append(current_lr)
        
        # Detect issues
        if total_norm > 100:
            print(f"\n⚠️  WARNING: Large gradient detected! Norm={total_norm:.2f} at Epoch {epoch}, Batch {batch_idx}")
        elif total_norm < 1e-7:
            print(f"\n⚠️  WARNING: Vanishing gradient detected! Norm={total_norm:.2e} at Epoch {epoch}, Batch {batch_idx}")


    def plot_training_curves(self):
        """
        Plot and save training/validation loss curves
        """
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_plot_path = os.path.join(self.log_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path, dpi=150)
        plt.close()
        
        print(f"✅ Loss curve saved to: {loss_plot_path}")

    def save_training_logs(self):
        """
        Save all training metrics to CSV files
        """
        # 1. Save gradient statistics
        gradient_csv_path = os.path.join(self.log_dir, 'gradient_statistics.csv')
        with open(gradient_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'epoch', 'batch', 'encoder_grad_norm', 'decoder_grad_norm', 
                'total_grad_norm', 'encoder_grad_max', 'decoder_grad_max',
                'encoder_grad_min', 'decoder_grad_min', 'loss', 'learning_rate'
            ])
            # Write data
            num_samples = len(self.gradient_stats['epoch'])
            for i in range(num_samples):
                writer.writerow([
                    self.gradient_stats['epoch'][i],
                    self.gradient_stats['batch'][i],
                    self.gradient_stats['encoder_grad_norm'][i],
                    self.gradient_stats['decoder_grad_norm'][i],
                    self.gradient_stats['total_grad_norm'][i],
                    self.gradient_stats['encoder_grad_max'][i],
                    self.gradient_stats['decoder_grad_max'][i],
                    self.gradient_stats['encoder_grad_min'][i],
                    self.gradient_stats['decoder_grad_min'][i],
                    self.gradient_stats['loss'][i],
                    self.gradient_stats['learning_rate'][i]
                ])
        
        print(f"✅ Gradient statistics saved to: {gradient_csv_path}")
        # 2. Save epoch-level metrics
        epoch_csv_path = os.path.join(self.log_dir, 'epoch_metrics.csv')
        with open(epoch_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_wer'])
            
            num_epochs = len(self.train_losses)
            for i in range(num_epochs):
                val_wer = self.val_wers[i] if i < len(self.val_wers) else None
                writer.writerow([
                    i + 1,
                    self.train_losses[i],
                    self.val_losses[i] if i < len(self.val_losses) else None,
                    val_wer
                ])
        
        print(f"✅ Epoch metrics saved to: {epoch_csv_path}")
        
        # 3. Save summary statistics
        summary_path = os.path.join(self.log_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Epochs: {len(self.train_losses)}\n")
            f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n")
            f.write(f"Best Validation WER: {self.best_val_wer:.4f}\n\n")
            
            f.write("GRADIENT STATISTICS:\n")
            f.write("-"*80 + "\n")
            if len(self.gradient_stats['total_grad_norm']) > 0:
                grad_norms = np.array(self.gradient_stats['total_grad_norm'])
                f.write(f"Mean Gradient Norm: {grad_norms.mean():.4f}\n")
                f.write(f"Max Gradient Norm: {grad_norms.max():.4f}\n")
                f.write(f"Min Gradient Norm: {grad_norms.min():.4f}\n")
                f.write(f"Std Gradient Norm: {grad_norms.std():.4f}\n\n")
                
                # Count gradient issues
                exploding_count = np.sum(grad_norms > 100)
                vanishing_count = np.sum(grad_norms < 1e-7)
                
                f.write(f"Exploding Gradient Instances (norm > 100): {exploding_count}\n")
                f.write(f"Vanishing Gradient Instances (norm < 1e-7): {vanishing_count}\n")
            
            f.write("="*80 + "\n")
        
        print(f"✅ Training summary saved to: {summary_path}")
        self.plot_training_curves()


    



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
                targets_list = batch['targets']  #List of sequences
                batch_size = images.size(0)
                
                # Forward pass through encoder
                encoder_output = self.encoder(images)
                annotations = reshape_fcn_output(encoder_output)
                
                if use_beam_search:
                    #Use beam search for prediction
                    predicted_sequences = []
                    for i in range(batch_size):
                        single_annotations = annotations[i:i+1]
                        
                        pred_seq, _ = self.decoder.decode_beam_search(
                            annotations=single_annotations,
                            start_token=self.start_idx,
                            end_token=self.end_idx,
                            max_len=150,
                            beam_width=beam_width
                        )
                        predicted_sequences.append(pred_seq)
                    
                    # Compute loss using teacher forcing (for monitoring)
                    outputs_list, attentions_list = self.decoder(
                        annotations,
                        targets_list,
                        teacher_forcing_ratio=1.0
                    )
                    
                    #Compute loss from individual sequences
                    total_loss_batch = 0.0
                    for i in range(batch_size):
                        outputs_single = outputs_list[i]
                        target_single = targets_list[i].to(self.device)
                        target_shifted = target_single[1:]
                        loss_single = self.criterion(outputs_single, target_shifted)
                        total_loss_batch += loss_single
                    
                    loss = total_loss_batch / batch_size
                    val_loss += loss.item()
                    
                    # Compute WER using beam search predictions
                    if compute_wer_flag:
                        references = []
                        hypotheses = []
                        for i in range(batch_size):
                            # Reference from targets_list
                            ref = targets_list[i].cpu().numpy().tolist()
                            # Remove START token
                            if ref[0] == self.start_idx:
                                ref = ref[1:]
                            # Stop at END
                            if self.end_idx in ref:
                                end_pos = ref.index(self.end_idx)
                                ref = ref[:end_pos]
                            ref = [token for token in ref if token != self.pad_idx]
                            
                            # Hypothesis: beam search prediction
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
                    #Teacher forcing mode
                    outputs_list, attentions_list = self.decoder(
                        annotations,
                        targets_list,
                        teacher_forcing_ratio=1.0
                    )
                    
                    # Compute loss from individual sequences
                    total_loss_batch = 0.0
                    for i in range(batch_size):
                        outputs_single = outputs_list[i]
                        target_single = targets_list[i].to(self.device)
                        target_shifted = target_single[1:]
                        loss_single = self.criterion(outputs_single, target_shifted)
                        total_loss_batch += loss_single
                    
                    loss = total_loss_batch / batch_size
                    val_loss += loss.item()
                    
                    # Compute WER from individual sequences
                    if compute_wer_flag:
                        references = []
                        hypotheses = []
                        for i in range(batch_size):
                            # Reference from targets_list
                            ref = targets_list[i].cpu().numpy().tolist()
                            # Remove START token
                            if ref[0] == self.start_idx:
                                ref = ref[1:]
                            # Stop at END
                            if self.end_idx in ref:
                                end_pos = ref.index(self.end_idx)
                                ref = ref[:end_pos]
                            ref = [token for token in ref if token != self.pad_idx]
                            
                            # Hypothesis from predictions
                            predicted_indices = outputs_list[i].argmax(dim=1).cpu().numpy().tolist()
                            
                            # Match reference length during teacher forcing
                            hyp = predicted_indices[:len(ref)]
                            
                            references.append(ref)
                            hypotheses.append(hyp)
                        
                        batch_metrics = batch_wer(references, hypotheses, 
                                                self.pad_idx, self.start_idx, self.end_idx)
                        
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
                    current_wer = (total_substitutions + total_deletions + total_insertions) / total_words
                    postfix['avg_wer'] = f'{current_wer:.4f}'
                    postfix['mode'] = 'beam_search' if use_beam_search else 'teacher_forcing'
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
            
            with open('/kaggle/input/mathwriting-symbols/ProcessedMathWrittingSymbols/idx2word.pkl', 'rb') as f:
                idx2word = pickle.load(f)
            
            train_loss = self.train_epoch_minimal_debug(
                train_loader, epoch, 
                teacher_forcing_ratio=teacher_forcing_ratio,
                idx2word=idx2word
            )

            # ✅ Use teacher forcing for validation (faster)
            compute_wer_flag = (epoch % compute_wer_every == 0)
            val_loss, val_wer = self.validate(
                val_loader, 
                compute_wer_flag=compute_wer_flag,
                use_beam_search=False,  # ✅ Removed beam search during training
                beam_width=5
            )
            
            self.scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start_time
            
            # Check if this is the best model
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

            # Print gradient statistics for this epoch
            epoch_grad_norms = [
                norm for e, norm in zip(self.gradient_stats['epoch'], self.gradient_stats['total_grad_norm'])
                if e == epoch
            ]
            if epoch_grad_norms:
                print(f"Gradient Norm - Mean: {np.mean(epoch_grad_norms):.4f}, "
                      f"Max: {np.max(epoch_grad_norms):.4f}, "
                      f"Min: {np.min(epoch_grad_norms):.4f}")
        
        print(f"\nTraining completed!")
        
        # ✅ NEW: Save all logs after training
        print("\n" + "="*80)
        print("SAVING TRAINING LOGS...")
        print("="*80)
        self.save_training_logs()
    
    
    def train_epoch_minimal_debug(self, train_loader, epoch, teacher_forcing_ratio=1.0, idx2word=None):
        """
        Training loop processing each sequence individually (no padding).
        """
        self.encoder.train()
        self.decoder.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)  # (batch, 5, H, W)
            targets_list = batch['targets']  # List of variable-length tensors
            labels = batch['labels']
            
            batch_size = images.size(0)
            
            #Encode entire batch together
            encoder_output = self.encoder(images)  # (batch, 128, H, W)
            annotations = reshape_fcn_output(encoder_output)  # (batch, L, 128)
            
            #Decode each sequence individually
            outputs_list, attentions_list = self.decoder(
                annotations,
                targets_list,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            total_loss = 0.0
            for i in range(batch_size):
                outputs_single = outputs_list[i]  # (seq_len-1, K)
                target_single = targets_list[i].to(self.device)  # (seq_len,)
                
                # Target for loss: skip START token
                target_shifted = target_single[1:]  # (seq_len-1,)
                
                # Compute NLL loss for this sequence
                loss_single = self.criterion(outputs_single, target_shifted)
                total_loss += loss_single
            
            # Average loss over batch
            loss = total_loss / batch_size


            if batch_idx % 100 == 0:
                for i in range(min(batch_size, 2)):  # Print first 2 samples
                    target_single = targets_list[i].cpu().numpy().tolist()
                    outputs_single = outputs_list[i]
                    
                    # Get predictions
                    predicted_indices = outputs_single.argmax(dim=1).cpu().numpy().tolist()
                    
                    # Reference (skip START)
                    ref = target_single[1:]
                    if self.end_idx in ref:
                        end_pos = ref.index(self.end_idx)
                        ref = ref[:end_pos]
                    
                    # Hypothesis
                    hyp = predicted_indices[:len(ref)]
                    
                    # Compute WER
                    wer = compute_wer_basic(ref, hyp)
                    
                    print(f"Epoch={epoch} Batch={batch_idx} Sample={i} | "
                        f"Loss={loss.item():.4f} WER={wer:.4f} | "
                        f"SeqLen={len(ref)}")
                    
                    if idx2word is not None:
                        target_symbols = []
                        for t in ref:
                            if t == self.pad_idx:
                                target_symbols.append('[PAD]')
                            elif t == self.end_idx:
                                target_symbols.append('[END]')
                            elif t == self.start_idx:
                                target_symbols.append('[START]')
                            else:
                                target_symbols.append(idx2word.get(int(t), f'<UNK:{t}>'))
                        
                        predicted_symbols = []
                        for t in hyp:
                            if t == self.pad_idx:
                                predicted_symbols.append('[PAD]')
                            elif t == self.end_idx:
                                predicted_symbols.append('[END]')
                            elif t == self.start_idx:
                                predicted_symbols.append('[START]')
                            else:
                                predicted_symbols.append(idx2word.get(int(t), f'<UNK:{t}>'))
                        
                        print(f"Target:    {' '.join(target_symbols)}")
                        print(f"Predicted: {' '.join(predicted_symbols)}\n")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.log_gradients(epoch, batch_idx, loss.item())
                        
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=5.0)
            # torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            postfix = {
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'grad_norm': f'{self.gradient_stats["total_grad_norm"][-1]:.4f}'
            }
            pbar.set_postfix(postfix)
        
        avg_epoch_loss = epoch_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        return avg_epoch_loss
            
    def _print_graph_trace(self, grad_fn, depth=0, max_depth=10):
        """
        Recursively print the computation graph structure
        Args:
            grad_fn: The gradient function node
            depth: Current depth in the graph
            max_depth: Maximum depth to traverse
        """
        if grad_fn is None or depth > max_depth:
            return
        
        indent = "  " * depth
        print(f"{indent}├─ {type(grad_fn).__name__}")
        
        # Print metadata if available
        if hasattr(grad_fn, 'metadata'):
            print(f"{indent}│  metadata: {grad_fn.metadata}")
        
        # Traverse to next functions
        if hasattr(grad_fn, 'next_functions'):
            for next_fn, _ in grad_fn.next_functions:
                if next_fn is not None:
                    self._print_graph_trace(next_fn, depth + 1, max_depth)

    def evaluate_test_set(self, test_loader, beam_width=5, save_results=True):
        """
        Evaluate model on test set using beam search
        """
        self.encoder.eval()
        self.decoder.eval()
        
        # Load idx2word for visualization
        with open('/kaggle/input/mathwriting-symbols/ProcessedMathWrittingSymbols/idx2word.pkl', 'rb') as f:
            idx2word = pickle.load(f)
        
        # WER tracking
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
                targets_list = batch['targets']  # ✅ FIXED: List of sequences
                labels = batch['labels']
                filenames = batch['filenames']
                batch_size = images.size(0)
            
                # Forward pass through encoder
                encoder_output = self.encoder(images)
                annotations = reshape_fcn_output(encoder_output)
                
                # Use beam search for each sample
                for i in range(batch_size):
                    single_annotations = annotations[i:i+1]
                    
                    pred_seq, beam_score = self.decoder.decode_beam_search(
                        annotations=single_annotations,
                        start_token=self.start_idx,
                        end_token=self.end_idx,
                        max_len=150,
                        beam_width=beam_width
                    )
                
                    # Convert beam_score to float
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
                
                    # ✅ FIXED: Get reference from targets_list
                    ref = targets_list[i].cpu().numpy().tolist()
                    # Remove START token
                    if ref[0] == self.start_idx:
                        ref = ref[1:]
                    # Stop at END
                    if self.end_idx in ref:
                        end_pos = ref.index(self.end_idx)
                        ref = ref[:end_pos]
                    ref = [token for token in ref if token != self.pad_idx]
                    
                    # Get hypothesis
                    hyp = pred_seq
                    if len(hyp) > 0 and hyp[0] == self.start_idx:
                        hyp = hyp[1:]
                    if len(hyp) > 0 and hyp[-1] == self.end_idx:
                        hyp = hyp[:-1]
                    hyp = [token for token in hyp if token != self.pad_idx]
                    
                    # Compute WER
                    sample_metrics = compute_wer_detailed(ref, hyp)
                    
                    total_substitutions += sample_metrics['substitutions']
                    total_deletions += sample_metrics['deletions']
                    total_insertions += sample_metrics['insertions']
                    total_correct += sample_metrics['correct']
                    total_words += sample_metrics['total']
                
                    # Convert to symbols
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
                
                # Update progress bar
                if total_words > 0:
                    current_wer = (total_substitutions + total_deletions + total_insertions) / total_words
                    pbar.set_postfix({
                        'avg_wer': f'{current_wer:.4f}',
                        'samples': len(results)
                    })

        # Calculate final metrics
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

        # Print summary
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

        # Save results to file
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
        
                # Write detailed results for each sample
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
    DATA_DIR = "/kaggle/input/mathwriting-symbols/ProcessedMathWrittingSymbols"
    TRAIN_CSV = os.path.join(DATA_DIR, 'train_database.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'test_database.csv')  # For later use
    WORD2IDX_PATH = "/kaggle/input/mathwriting-symbols/ProcessedMathWrittingSymbols/word2idx.pkl"
    IDX2WORD_PATH = "/kaggle/input/mathwriting-symbols/ProcessedMathWrittingSymbols/idx2word.pkl"
    BASE_IMAGE_DIR = DATA_DIR  # Base directory
    
    train_image_dir = os.path.join(BASE_IMAGE_DIR, 'train')
    test_image_dir = os.path.join(BASE_IMAGE_DIR, 'test')  # Changed from 'val' to 'test'
    
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
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
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
        base_image_dir=train_image_dir,
        device=device,
        transform=None,
        #subset_size=100# Uncomment to limit dataset size
    )
    
    # ✅ Create validation dataset using test split
    val_dataset = MathExpressionDataset(
        csv_path=TEST_CSV,  # Using test as validation
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=test_image_dir,
        device=device,
        transform=None,
        #subset_size=10
    )
    test_dataset = MathExpressionDataset(
        csv_path=TEST_CSV,
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=test_image_dir,
        device=device,
        transform=None,
        #subset_size=10  # Use full test set
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
    # ✅ Train model (NO beam search during training)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        teacher_forcing_schedule=teacher_forcing_schedule_constant,
        resume_from=RESUME_FROM,
        compute_wer_every=1
    )
    
    # ✅ After training is complete, evaluate on test set with beam search
    print("\n" + "="*80)
    print("TRAINING COMPLETE - EVALUATING ON TEST SET WITH BEAM SEARCH")
    print("="*80 + "\n")
    
    # Load best checkpoint
    best_checkpoint = os.path.join(CHECKPOINT_DIR, 'checkpoint_best.pth')
    if os.path.exists(best_checkpoint):
        print(f"Loading best checkpoint: {best_checkpoint}")
        trainer.load_checkpoint(best_checkpoint)
    else:
        print("Warning: Best checkpoint not found, using current model state")
    
    # Evaluate with beam search
    test_metrics, test_results = trainer.evaluate_test_set(
        test_loader=test_loader,
        beam_width=5,  # Can use larger beam width for final evaluation
        save_results=True
    )
    
    print("\n" + "="*80)
    print("ALL EVALUATION COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
