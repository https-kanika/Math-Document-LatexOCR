"""
Evaluation script for Math OCR model (WatcherFCN + ParserGRUDecoder)
Computes CER, WER, Exact Match metrics and generates visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import editdistance
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Dataset Class (copied from your mathOCR.py)
# ============================================================================

def get_directional_kernels():
    k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernels = [
        k,
        np.rot90(k, 1),
        np.rot90(k, 2),
        np.rot90(k, 3),
        np.fliplr(k),
        np.flipud(k),
        np.fliplr(np.rot90(k, 1)),
        np.flipud(np.rot90(k, 3)),
    ]
    return kernels

def get_directional_maps(image):
    kernels = get_directional_kernels()
    edge_maps = [cv2.filter2D(image, -1, kern) for kern in kernels]
    edge_maps = [(em.astype(np.float32) / 255.0) for em in edge_maps]
    edge_maps = [np.clip(em, 0, 1) for em in edge_maps]
    return np.stack(edge_maps, axis=0)

class MathEquation9ChDataset(Dataset):
    def __init__(self, csv_file, dataset_root, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.dataset_root = dataset_root
        self.transform = transform
        
        self.data_frame['image_path'] = self.data_frame['image_path'].apply(
            lambda x: os.path.normpath(x).replace('\\', '/')
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        relative_img_path = self.data_frame.iloc[idx]['image_path']
        img_full_path = os.path.join(self.dataset_root, relative_img_path)
        img_full_path = os.path.normpath(img_full_path).replace('\\', '/')
        
        image = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_full_path}")
        image = image.astype(np.float32) / 255.0
        H, W = image.shape
        
        channels = np.zeros((9, H, W), dtype=np.float32)
        channels[0] = image
        channels[1:] = get_directional_maps(image)
        
        label = self.data_frame.iloc[idx]['normalized_label']
        sample = {'image': torch.tensor(channels, dtype=torch.float32), 'label': label}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


# ============================================================================
# Model Classes (copied from your mathOCR.py)
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class WatcherFCN(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        return x

class CoverageAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, coverage_dim):
        super().__init__()
        self.W_a = nn.Linear(decoder_dim, attention_dim)
        self.U_a = nn.Linear(encoder_dim, attention_dim)
        self.U_f = nn.Linear(coverage_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden, coverage):
        Wh = self.W_a(decoder_hidden).unsqueeze(1)
        Ua = self.U_a(encoder_outputs)
        Uf = self.U_f(coverage)
        att = torch.tanh(Wh + Ua + Uf)
        scores = self.v(att).squeeze(-1)
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(encoder_outputs * alpha.unsqueeze(-1), dim=1)
        return context, alpha

class ParserGRUDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim=512, embed_dim=256, decoder_dim=256, attention_dim=256, coverage_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.attention = CoverageAttention(encoder_dim, decoder_dim, attention_dim, coverage_dim)
        self.fc = nn.Linear(decoder_dim + encoder_dim, vocab_size)

    def forward(self, encoder_outputs, targets, max_len):
        batch_size, L, encoder_dim = encoder_outputs.size()
        device = encoder_outputs.device
        coverage = torch.zeros(batch_size, L, 1, device=device)
        inputs = torch.full((batch_size,), 1, dtype=torch.long, device=device)
        hidden = torch.zeros(batch_size, 256, device=device)
        outputs = []
        
        for t in range(max_len):
            embedded = self.embedding(inputs)
            context, alpha = self.attention(encoder_outputs, hidden, coverage)
            gru_input = torch.cat([embedded, context], dim=1)
            hidden = self.gru(gru_input, hidden)
            output = self.fc(torch.cat([hidden, context], dim=1))
            outputs.append(output)
            
            if targets is not None and t < targets.size(1):
                inputs = targets[:, t]
            else:
                inputs = output.argmax(dim=1)
            coverage = coverage + alpha.unsqueeze(-1)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs


# ============================================================================
# Evaluator Class
# ============================================================================

class MathOCREvaluator:
    def __init__(self, watcher, decoder, vocab, device, max_len=128):
        self.watcher = watcher
        self.decoder = decoder
        self.vocab = vocab
        self.device = device
        self.max_len = max_len
        self.pad_idx = vocab.index('<PAD>')
        self.sos_idx = vocab.index('<SOS>')
        self.eos_idx = vocab.index('<EOS>')
        self.idx2char = {idx: ch for idx, ch in enumerate(vocab)}
        
    def encode_label(self, label, max_len=None):
        if max_len is None:
            max_len = self.max_len
        tokens = [self.sos_idx] + [self.vocab.index(ch) for ch in label] + [self.eos_idx]
        if len(tokens) < max_len:
            tokens += [self.pad_idx] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens
    
    def decode_sequence(self, indices):
        chars = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.sos_idx]:
                chars.append(self.idx2char[idx])
        return ''.join(chars)
    
    def compute_cer(self, reference, hypothesis):
        if len(reference) == 0:
            return 0.0 if len(hypothesis) == 0 else 1.0
        return editdistance.eval(reference, hypothesis) / len(reference)
    
    def compute_wer(self, reference, hypothesis):
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        if len(ref_tokens) == 0:
            return 0.0 if len(hyp_tokens) == 0 else 1.0
        return editdistance.eval(ref_tokens, hyp_tokens) / len(ref_tokens)
    
    def validate(self, val_loader):
        self.watcher.eval()
        self.decoder.eval()
        
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        total_loss = 0
        all_refs = []
        all_preds = []
        
        print("\n" + "="*60)
        print("VALIDATION")
        print("="*60)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels_str = batch['label']
                
                labels = torch.tensor(
                    [self.encode_label(lbl) for lbl in labels_str],
                    dtype=torch.long,
                    device=self.device
                )
                
                watcher_output = self.watcher(images)
                batch_size, channels, height, width = watcher_output.shape
                encoder_outputs = watcher_output.permute(0, 2, 3, 1).reshape(
                    batch_size, height * width, channels
                )
                
                outputs = self.decoder(encoder_outputs, labels, self.max_len)
                
                outputs_flat = outputs.view(-1, outputs.size(-1))
                labels_flat = labels.view(-1)
                loss = criterion(outputs_flat, labels_flat)
                total_loss += loss.item()
                
                predicted_indices = outputs.argmax(dim=-1).cpu().numpy()
                
                for i in range(batch_size):
                    pred_str = self.decode_sequence(predicted_indices[i])
                    ref_str = labels_str[i]
                    all_preds.append(pred_str)
                    all_refs.append(ref_str)
        
        avg_loss = total_loss / len(val_loader)
        
        cer_scores = [self.compute_cer(r, p) for r, p in zip(all_refs, all_preds)]
        wer_scores = [self.compute_wer(r, p) for r, p in zip(all_refs, all_preds)]
        exact_matches = [1.0 if r == p else 0.0 for r, p in zip(all_refs, all_preds)]
        
        metrics = {
            'loss': avg_loss,
            'CER': np.mean(cer_scores),
            'CER_std': np.std(cer_scores),
            'WER': np.mean(wer_scores),
            'WER_std': np.std(wer_scores),
            'Exact_Match': np.mean(exact_matches) * 100,
            'num_samples': len(all_refs)
        }
        
        return metrics, all_refs, all_preds
    
    def analyze_errors(self, references, predictions, top_k=20):
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        errors = []
        for ref, pred in zip(references, predictions):
            if ref != pred:
                cer = self.compute_cer(ref, pred)
                errors.append({
                    'reference': ref,
                    'prediction': pred,
                    'cer': cer,
                    'ref_len': len(ref),
                    'pred_len': len(pred),
                    'len_diff': abs(len(ref) - len(pred))
                })
        
        if not errors:
            print("No errors found! Perfect accuracy!")
            return []
        
        errors_sorted = sorted(errors, key=lambda x: x['cer'], reverse=True)
        
        print(f"\nTotal errors: {len(errors)} out of {len(references)} samples")
        print(f"Error rate: {len(errors)/len(references)*100:.2f}%")
        
        print(f"\n{'='*60}")
        print(f"Top {min(top_k, len(errors))} Worst Predictions:")
        print(f"{'='*60}")
        
        for i, err in enumerate(errors_sorted[:top_k], 1):
            print(f"\n{i}. CER: {err['cer']:.3f}")
            print(f"   Reference: {err['reference'][:100]}")
            print(f"   Predicted: {err['prediction'][:100]}")
            print(f"   Lengths: ref={err['ref_len']}, pred={err['pred_len']}")
        
        cer_values = [e['cer'] for e in errors]
        len_diffs = [e['len_diff'] for e in errors]
        
        print(f"\n{'='*60}")
        print("Error Statistics:")
        print(f"{'='*60}")
        print(f"Mean CER (errors only): {np.mean(cer_values):.4f}")
        print(f"Median CER: {np.median(cer_values):.4f}")
        print(f"Mean length difference: {np.mean(len_diffs):.2f} characters")
        
        return errors
    
    def visualize_predictions(self, val_loader, num_samples=10, save_dir='./evaluation_results'):
        os.makedirs(save_dir, exist_ok=True)
        
        self.watcher.eval()
        self.decoder.eval()
        
        print("\n" + "="*60)
        print(f"VISUALIZING {num_samples} SAMPLE PREDICTIONS")
        print("="*60)
        
        sample_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if sample_count >= num_samples:
                    break
                
                images = batch['image'].to(self.device)
                labels_str = batch['label']
                
                watcher_output = self.watcher(images)
                batch_size, channels, height, width = watcher_output.shape
                encoder_outputs = watcher_output.permute(0, 2, 3, 1).reshape(
                    batch_size, height * width, channels
                )
                
                labels = torch.tensor(
                    [self.encode_label(lbl) for lbl in labels_str],
                    dtype=torch.long,
                    device=self.device
                )
                
                outputs = self.decoder(encoder_outputs, labels, self.max_len)
                predicted_indices = outputs.argmax(dim=-1).cpu().numpy()
                
                for i in range(min(batch_size, num_samples - sample_count)):
                    img = images[i][0].cpu().numpy()
                    
                    ref_str = labels_str[i]
                    pred_str = self.decode_sequence(predicted_indices[i])
                    cer = self.compute_cer(ref_str, pred_str)
                    
                    fig, ax = plt.subplots(figsize=(15, 5))
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    
                    title = f"Sample {sample_count + 1}\n"
                    title += f"Reference: {ref_str[:80]}...\n" if len(ref_str) > 80 else f"Reference: {ref_str}\n"
                    title += f"Predicted: {pred_str[:80]}...\n" if len(pred_str) > 80 else f"Predicted: {pred_str}\n"
                    title += f"CER: {cer:.4f}"
                    
                    color = 'green' if cer < 0.1 else 'orange' if cer < 0.3 else 'red'
                    ax.set_title(title, fontsize=10, loc='left', color=color)
                    
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(save_dir, f'sample_{sample_count+1:03d}.png'),
                        dpi=150,
                        bbox_inches='tight'
                    )
                    plt.close()
                    
                    sample_count += 1
                    
                    if sample_count >= num_samples:
                        break
        
        print(f"Saved {sample_count} visualizations to {save_dir}")
    
    def plot_metrics(self, metrics, save_path='./evaluation_results/metrics.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metric_names = ['CER (%)', 'WER (%)', 'Exact Match (%)']
        metric_values = [
            metrics['CER'] * 100,
            metrics['WER'] * 100,
            metrics['Exact_Match']
        ]
        colors = ['steelblue', 'coral', 'seagreen']
        
        for ax, name, value, color in zip(axes, metric_names, metric_values, colors):
            ax.bar([name.split()[0]], [value], alpha=0.8, color=color, width=0.5)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            ax.text(0, value + 2, f'{value:.2f}%', ha='center', fontsize=12, fontweight='bold')
        
        plt.suptitle('Model Evaluation Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nMetrics plot saved to {save_path}")
        plt.show()


# ============================================================================
# Main Evaluation Function
# ============================================================================

def run_evaluation(
    model_path,
    dataset_root,
    val_csv,
    vocab,
    device,
    batch_size=8,
    num_visualize=20,
    save_dir='./evaluation_results'
):
    """
    Run comprehensive evaluation
    
    Args:
        model_path: Path to saved model checkpoint (.pth file)
        dataset_root: Root directory of dataset
        val_csv: Path to validation CSV file
        vocab: List of vocabulary tokens
        device: torch device (cuda/cpu)
        batch_size: Batch size for evaluation
        num_visualize: Number of samples to visualize
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("MATH OCR MODEL EVALUATION")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading validation dataset from: {val_csv}")
    val_dataset = MathEquation9ChDataset(val_csv, dataset_root)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Initialize models
    print("\nInitializing models...")
    watcher = WatcherFCN(in_channels=9).to(device)
    decoder = ParserGRUDecoder(vocab_size=len(vocab)).to(device)
    
    # Load checkpoint
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    watcher.load_state_dict(checkpoint['watcher_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  Training loss: {checkpoint['loss']:.4f}")
    
    # Initialize evaluator
    evaluator = MathOCREvaluator(watcher, decoder, vocab, device)
    
    # 1. Validation
    metrics, all_refs, all_preds = evaluator.validate(val_loader)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 2. Error analysis
    errors = evaluator.analyze_errors(all_refs, all_preds, top_k=20)
    
    # 3. Visualize predictions
    evaluator.visualize_predictions(val_loader, num_samples=num_visualize, save_dir=save_dir)
    
    # 4. Plot metrics
    evaluator.plot_metrics(metrics, save_path=os.path.join(save_dir, 'metrics.png'))
    
    # 5. Save detailed results
    results_df = pd.DataFrame({
        'reference': all_refs,
        'prediction': all_preds,
        'cer': [evaluator.compute_cer(r, p) for r, p in zip(all_refs, all_preds)],
        'wer': [evaluator.compute_wer(r, p) for r, p in zip(all_refs, all_preds)],
        'exact_match': [1 if r == p else 0 for r, p in zip(all_refs, all_preds)]
    })
    results_csv = os.path.join(save_dir, 'detailed_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")
    
    # 6. Generate summary report
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MATH OCR MODEL - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("VALIDATION METRICS\n")
        f.write("-"*80 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\n\nERROR ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total errors: {len(errors)}\n")
        f.write(f"Error rate: {len(errors)/len(all_refs)*100:.2f}%\n")
        
        if errors:
            cer_values = [e['cer'] for e in errors]
            f.write(f"Mean CER (errors only): {np.mean(cer_values):.4f}\n")
            f.write(f"Median CER: {np.median(cer_values):.4f}\n")
    
    print(f"Evaluation report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {save_dir}")
    print(f"\nFINAL SUMMARY:")
    print(f"  CER: {metrics['CER']*100:.2f}%")
    print(f"  WER: {metrics['WER']*100:.2f}%")
    print(f"  Exact Match: {metrics['Exact_Match']:.2f}%")
    print(f"  Total Errors: {len(errors)}/{len(all_refs)}")
    
    return metrics, errors


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATASET_ROOT = r'C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessMathwritting-exercpt'
    VAL_CSV = os.path.join(DATASET_ROOT, 'val_database.csv')
    MODEL_PATH = 'best_model.pth'  # or 'final_model.pth'
    
    # Build vocabulary (same as in training)
    csv_files = ['train_database.csv', 'val_database.csv', 'test_database.csv']
    all_labels = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(DATASET_ROOT, csv_file))
        all_labels.extend(df['normalized_label'].astype(str).tolist())
    
    from collections import Counter
    special_tokens = ['<PAD>', '<SOS>', '<EOS>']
    char_counter = Counter()
    for label in all_labels:
        char_counter.update(list(label))
    
    vocab = special_tokens + sorted(char_counter.keys())
    print(f"Vocabulary size: {len(vocab)}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run evaluation
    metrics, errors = run_evaluation(
        model_path=MODEL_PATH,
        dataset_root=DATASET_ROOT,
        val_csv=VAL_CSV,
        vocab=vocab,
        device=device,
        batch_size=8,
        num_visualize=20,
        save_dir='./evaluation_results'
    )