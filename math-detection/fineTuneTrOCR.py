#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import (
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator,
    ViTImageProcessor,
    RobertaTokenizer,
    TrOCRProcessor
)
import re
from collections import Counter
import pickle

# ============================================================================
# VOCABULARY CREATION FUNCTIONS
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


def build_vocab_from_dataframe(df):
    """Build vocabulary from dataframe with 'text' column"""
    token_counter = Counter()
    
    for text in df['text']:
        cleaned = clean_latex_label(str(text))
        tokens = tokenize_latex(cleaned)
        token_counter.update(tokens)
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    sorted_tokens = [token for token, count in token_counter.most_common()]
    all_tokens = special_tokens + sorted_tokens
    
    token2idx = {token: idx for idx, token in enumerate(all_tokens)}
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    return token2idx, idx2token, token_counter


def save_vocabulary(token2idx, idx2token, output_dir='./vocab/'):
    """Save vocabulary files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pickle files
    with open(os.path.join(output_dir, 'token2idx.pkl'), 'wb') as f:
        pickle.dump(token2idx, f)
    
    with open(os.path.join(output_dir, 'idx2token.pkl'), 'wb') as f:
        pickle.dump(idx2token, f)
    
    # Save vocab.txt
    with open(os.path.join(output_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for token, idx in sorted(token2idx.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{token}\n")
    
    print(f"Vocabulary saved to {output_dir}")
    print(f"Total unique tokens: {len(token2idx)}")


def load_vocabulary(vocab_dir='./vocab/'):
    """Load vocabulary from saved files"""
    with open(os.path.join(vocab_dir, 'token2idx.pkl'), 'rb') as f:
        token2idx = pickle.load(f)
    
    with open(os.path.join(vocab_dir, 'idx2token.pkl'), 'rb') as f:
        idx2token = pickle.load(f)
    
    return token2idx, idx2token


# ============================================================================
# CUSTOM TOKENIZER FOR LATEX
# ============================================================================

class LaTeXTokenizer:
    """Custom tokenizer for LaTeX expressions"""
    
    def __init__(self, token2idx, idx2token):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.vocab_size = len(token2idx)
        
        # Special token IDs
        self.pad_token_id = token2idx['<PAD>']
        self.cls_token_id = token2idx['<START>']  # CLS = START
        self.sep_token_id = token2idx['<END>']    # SEP = END
        self.unk_token_id = token2idx['<UNK>']
    
    def encode(self, text, max_length=490, padding='max_length'):
        """Encode text to token IDs"""
        cleaned = clean_latex_label(str(text))
        tokens = tokenize_latex(cleaned)
        
        # Convert tokens to IDs
        token_ids = [self.token2idx.get(token, self.unk_token_id) for token in tokens]
        
        # Add START and END tokens
        token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        
        # Padding or truncation
        if padding == 'max_length':
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        return token_ids
    
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
# CUSTOM DATASET
# ============================================================================

class LaTeXMathDataset(Dataset):
    def __init__(self, root_dir, df, image_processor, tokenizer, max_target_length=490):
        self.root_dir = root_dir
        self.df = df
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'].iloc[idx]
        text = self.df['text'].iloc[idx]
        
        # Process image
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Encode text using custom tokenizer
        labels = self.tokenizer.encode(text, max_length=self.max_target_length, padding='max_length')
        
        # Ignore PAD tokens in loss
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

# Load data
print("Loading data...")
df = pd.read_table('./data/train/caption.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
df['file_name'] = df['file_name'].apply(lambda x: x+'.jpg')
df = df.dropna()

df2 = pd.read_table('./data/2014/caption.txt', header=None)
df2.rename(columns={0: "file_name", 1: "text"}, inplace=True)
df2['file_name'] = df2['file_name'].apply(lambda x: x+'.jpg')
df2 = df2.dropna()

train_df = shuffle(df)
test_df = df2
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Build or load vocabulary
vocab_dir = './vocab/'
if not os.path.exists(os.path.join(vocab_dir, 'token2idx.pkl')):
    print("Building vocabulary from training data...")
    token2idx, idx2token, token_counter = build_vocab_from_dataframe(train_df)
    save_vocabulary(token2idx, idx2token, vocab_dir)
    
    # Print top tokens
    print("\nTop 20 most common tokens:")
    for token, count in token_counter.most_common(20):
        print(f"  {token}: {count}")
else:
    print("Loading existing vocabulary...")
    token2idx, idx2token = load_vocabulary(vocab_dir)

# Create custom tokenizer
latex_tokenizer = LaTeXTokenizer(token2idx, idx2token)
print(f"Vocabulary size: {latex_tokenizer.vocab_size}")

from transformers import TrOCRProcessor
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
image_processor = trocr_processor.image_processor  # This will have correct settings

# Create datasets
print("\nCreating datasets...")
train_dataset = LaTeXMathDataset(
    root_dir='./data/train/',
    df=train_df,
    image_processor=image_processor,
    tokenizer=latex_tokenizer
)

eval_dataset = LaTeXMathDataset(
    root_dir='./data/2014/',
    df=test_df,
    image_processor=image_processor,
    tokenizer=latex_tokenizer
)

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(eval_dataset)}")

# Load model
print("\nLoading model...")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1")

# Resize decoder embeddings to match vocabulary
model.decoder.resize_token_embeddings(latex_tokenizer.vocab_size)

# Configure model
model.config.decoder_start_token_id = latex_tokenizer.cls_token_id
model.config.pad_token_id = latex_tokenizer.pad_token_id
model.config.vocab_size = latex_tokenizer.vocab_size
model.config.eos_token_id = latex_tokenizer.sep_token_id
model.config.max_length = 490
model.config.early_stopping = True
model.config.num_beams = 10

# ✅ FIX: Also update generation_config explicitly
model.generation_config.decoder_start_token_id = latex_tokenizer.cls_token_id  # 1
model.generation_config.eos_token_id = latex_tokenizer.sep_token_id            # 2
model.generation_config.pad_token_id = latex_tokenizer.pad_token_id            # 0
model.generation_config.bos_token_id = latex_tokenizer.cls_token_id            # 1

# Training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    per_device_train_batch_size=64,  # Reduced for stability
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,   # Effective batch = 32
    fp16=True,
    learning_rate=2e-4,               # Adjusted for larger batch
    warmup_steps=500,
    weight_decay=0.01,
    output_dir="./checkpoint_latex_vocab/",
    logging_steps=2,
    save_steps=500,
    eval_steps=500,
    num_train_epochs=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    dataloader_num_workers=4,
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=image_processor,  # For saving config
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

# Check for checkpoints to resume
checkpoint_dir = "./checkpoint_latex_vocab/"
resume_from = None

if os.path.exists(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]), reverse=True)
        resume_from = checkpoints[0]
        print(f"Resuming from checkpoint: {resume_from}")

# Train
print("\n" + "="*50)
print("FINAL VERIFICATION BEFORE TRAINING")
print("="*50)
print(f"Model decoder_start_token_id: {model.config.decoder_start_token_id}")
print(f"Model generation decoder_start_token_id: {model.generation_config.decoder_start_token_id}")
print(f"Model eos_token_id: {model.config.eos_token_id}")
print(f"Model pad_token_id: {model.config.pad_token_id}")
print(f"Expected START: {latex_tokenizer.cls_token_id} (should be 1)")
print(f"Expected END: {latex_tokenizer.sep_token_id} (should be 2)")
print(f"Expected PAD: {latex_tokenizer.pad_token_id} (should be 0)")
print("="*50 + "\n")

# ONLY proceed if all match!
assert model.generation_config.decoder_start_token_id == 1, "Wrong start token!"
assert model.config.decoder_start_token_id == 1, "Wrong start token!"
assert model.config.eos_token_id == 2, "Wrong end token!"
assert model.config.pad_token_id == 0, "Wrong pad token!"

print("All token IDs verified correctly!\n")

print("\nStarting training...")
trainer.train(resume_from_checkpoint=resume_from)

# Save final model and tokenizer
print("\nSaving final model...")
final_output_dir = "./final_model_latex_vocab/"
model.save_pretrained(final_output_dir)
image_processor.save_pretrained(final_output_dir)

# Save custom tokenizer
save_vocabulary(token2idx, idx2token, os.path.join(final_output_dir, 'vocab'))

print("\nTraining complete!")
print(f"Model saved to: {final_output_dir}")

# Evaluate
print("\nEvaluating model on validation set...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")