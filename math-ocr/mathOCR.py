import pandas as pd
import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch

def get_directional_kernels():
    # 8 edge detection kernels: N, NE, E, SE, S, SW, W, NW
    k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) # Vertical
    kernels = [
        k,                              # S
        np.rot90(k, 1),                 # W
        np.rot90(k, 2),                 # N
        np.rot90(k, 3),                 # E
        np.fliplr(k),                   # SW
        np.flipud(k),                   # NE
        np.fliplr(np.rot90(k, 1)),      # NW
        np.flipud(np.rot90(k, 3)),      # SE
    ]
    return kernels

def get_directional_maps(image):
    kernels = get_directional_kernels()
    edge_maps = [cv2.filter2D(image, -1, kern) for kern in kernels]  # 8 edge maps
    # Normalize each map to [0, 1] and clip negative values
    edge_maps = [(em.astype(np.float32) / 255.0) for em in edge_maps]
    edge_maps = [np.clip(em, 0, 1) for em in edge_maps]
    return np.stack(edge_maps, axis=0)  # shape [8, H, W]

class MathEquation9ChDataset(Dataset):
    def __init__(self, csv_file, dataset_root, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.dataset_root = dataset_root
        self.transform = transform
        
        # Normalize image paths in the dataframe
        self.data_frame['image_path'] = self.data_frame['image_path'].apply(
            lambda x: os.path.normpath(x).replace('\\', '/')
        )

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        relative_img_path = self.data_frame.iloc[idx]['image_path']
        img_full_path = os.path.join(self.dataset_root, relative_img_path)
        # Normalize the full path as well
        img_full_path = os.path.normpath(img_full_path).replace('\\', '/')
        
        image = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_full_path}")
        image = image.astype(np.float32) / 255.0
        H, W = image.shape
        # 9 channel construction
        channels = np.zeros((9, H, W), dtype=np.float32)
        channels[0] = image  # Greyscale base
        channels[1:] = get_directional_maps(image)  # 8 directions
        label = self.data_frame.iloc[idx]['normalized_label']
        sample = {'image': torch.tensor(channels, dtype=torch.float32), 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

# Usage:
DATASET_ROOT = r'C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessMathwritting-exercpt'
TRAIN_CSV = os.path.join(DATASET_ROOT, 'train_database.csv')

# Let's first check if the CSV exists and print its contents
if os.path.exists(TRAIN_CSV):
    df = pd.read_csv(TRAIN_CSV)
    print("CSV file loaded successfully")
    print("Columns:", df.columns.tolist())
    print("\nFirst few image paths:")
    print(df['image_path'].head())
else:
    print(f"CSV file not found at {TRAIN_CSV}")

train_dataset = MathEquation9ChDataset(TRAIN_CSV, DATASET_ROOT)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

try:
    for batch in train_loader:
        images, labels = batch['image'], batch['label']
        print(f"\nBatch loaded successfully")
        print(f"Image tensor shape: {images.shape}")
        print("First few labels:", labels[:3])
        break
except Exception as e:
    print(f"\nError loading batch: {str(e)}")

import pandas as pd
from collections import Counter

# Load all labels from train/val/test CSVs
csv_files = [
    'train_database.csv',
    'val_database.csv',
    'test_database.csv'
]
DATASET_ROOT = r'C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessMathwritting-exercpt'

all_labels = []
for csv_file in csv_files:
    df = pd.read_csv(os.path.join(DATASET_ROOT, csv_file))
    all_labels.extend(df['normalized_label'].astype(str).tolist())

# Build character-level vocabulary
special_tokens = ['<PAD>', '<SOS>', '<EOS>']
char_counter = Counter()
for label in all_labels:
    char_counter.update(list(label))

vocab = special_tokens + sorted(char_counter.keys())
char2idx = {ch: idx for idx, ch in enumerate(vocab)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

print(f"Vocabulary size: {len(vocab)}")
print("First 20 tokens:", vocab[:20])

# Encode a label string to indices
def encode_label(label, max_len=128):
    tokens = [char2idx['<SOS>']] + [char2idx[ch] for ch in label] + [char2idx['<EOS>']]
    if len(tokens) < max_len:
        tokens += [char2idx['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

# Example usage
sample_label = all_labels[0]
encoded = encode_label(sample_label)
print("Original label:", sample_label)
print("Encoded:", encoded[:20])

# For your dataset class, you can add:
# label_indices = encode_label(label)
# sample = {'image': image_tensor, 'label': label_indices}
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Output: [batch, 512, H/16, W/16]

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.pool4(x)
        return x  # [batch, 512, H/16, W/16]

# Example usage:
model = WatcherFCN(in_channels=9)
dummy_input = torch.randn(2, 9, 480, 1600)
output = model(dummy_input)
print(output.shape)  # Should be [2, 512, 30, 100]
batch_size, channels, height, width = output.shape
encoder_outputs = output.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
# encoder_outputs: [batch, 3000, 512]


import torch
import torch.nn as nn
import torch.nn.functional as F

class CoverageAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, coverage_dim):
        super().__init__()
        self.W_a = nn.Linear(decoder_dim, attention_dim)
        self.U_a = nn.Linear(encoder_dim, attention_dim)
        self.U_f = nn.Linear(coverage_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden, coverage):
        # encoder_outputs: [batch, L, encoder_dim]
        # decoder_hidden: [batch, decoder_dim]
        # coverage: [batch, L, coverage_dim]
        Wh = self.W_a(decoder_hidden).unsqueeze(1)  # [batch, 1, att_dim]
        Ua = self.U_a(encoder_outputs)              # [batch, L, att_dim]
        Uf = self.U_f(coverage)                     # [batch, L, att_dim]
        att = torch.tanh(Wh + Ua + Uf)              # [batch, L, att_dim]
        scores = self.v(att).squeeze(-1)            # [batch, L]
        alpha = F.softmax(scores, dim=1)            # [batch, L]
        context = torch.sum(encoder_outputs * alpha.unsqueeze(-1), dim=1)  # [batch, encoder_dim]
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
        inputs = torch.full((batch_size,), 1, dtype=torch.long, device=device)  # <SOS> token index
        hidden = torch.zeros(batch_size, 256, device=device)
        outputs = []
        for t in range(max_len):
            embedded = self.embedding(inputs)  # [batch, embed_dim]
            context, alpha = self.attention(encoder_outputs, hidden, coverage)
            gru_input = torch.cat([embedded, context], dim=1)
            hidden = self.gru(gru_input, hidden)
            output = self.fc(torch.cat([hidden, context], dim=1))
            outputs.append(output)
            # Teacher forcing: use ground truth if available
            if targets is not None and t < targets.size(1):
                inputs = targets[:, t]
            else:
                inputs = output.argmax(dim=1)
            coverage = coverage + alpha.unsqueeze(-1)
        outputs = torch.stack(outputs, dim=1)  # [batch, max_len, vocab_size]
        return outputs

# Example usage:
# encoder_outputs: [batch, L, encoder_dim] (flatten FCN output to [batch, L, 512])
# targets: [batch, max_len] (token indices)
# decoder = ParserGRUDecoder(vocab_size=len(vocab))
# outputs = decoder(encoder_outputs, targets, max_len)


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Initialize models with proper configuration
watcher = WatcherFCN(in_channels=9)  # 9-channel input as defined in dataset
decoder = ParserGRUDecoder(vocab_size=len(vocab))  # vocab was defined in previous cell

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Move models to device
watcher = watcher.to(device)
decoder = decoder.to(device)

pad_idx = vocab.index('<PAD>')
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adadelta(list(watcher.parameters()) + list(decoder.parameters()))

num_epochs = 10
max_len = 128

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
best_loss = float('inf')

# Rest of the training code remains the same...

try:
    for epoch in range(num_epochs):
        watcher.train()
        decoder.train()
        total_loss = 0
        batch_count = 0
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            # Move batch to device
            images = batch['image'].to(device)
            labels = [encode_label(lbl, max_len) for lbl in batch['label']]
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            try:
                watcher_output = watcher(images)
                batch_size, channels, height, width = watcher_output.shape
                encoder_outputs = watcher_output.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

                outputs = decoder(encoder_outputs, labels, max_len)
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)

                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(list(watcher.parameters()) + list(decoder.parameters()), max_norm=5.0)
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                print(f"Error in batch: {str(e)}")
                continue

        # Calculate average loss
        avg_loss = total_loss / batch_count
        print(f"\nEpoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'watcher_state_dict': watcher.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nError during training: {str(e)}")
finally:
    # Save final model
    torch.save({
        'watcher_state_dict': watcher.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss / len(train_loader) if 'total_loss' in locals() else None,
    }, 'final_model.pth')