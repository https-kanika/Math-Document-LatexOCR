import pandas as pd
import pickle
import os
import argparse
from collections import Counter

def create_vocabulary(csv_path, output_dir):
    """
    Create vocabulary files from symbols dataset CSV
    
    Args:
        csv_path: Path to the symbols database CSV file
        output_dir: Directory to save vocabulary files
    """
    print(f"Reading CSV from: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get all unique labels
    all_labels = df['label'].dropna().unique().tolist()
    
    # Sort labels for consistency
    all_labels = sorted(all_labels)
    
    print(f"Found {len(all_labels)} unique symbols")
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    
    # Create vocabulary with special tokens first
    vocab = special_tokens + all_labels
    
    print(f"Total vocabulary size (including special tokens): {len(vocab)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save vocab.txt
    vocab_txt_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_txt_path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f"{token}\n")
    print(f"Saved vocab.txt to {vocab_txt_path}")
    
    # 2. Create word2idx mapping
    word2idx = {token: idx for idx, token in enumerate(vocab)}
    
    # Save word2idx.pkl
    word2idx_path = os.path.join(output_dir, "word2idx.pkl")
    with open(word2idx_path, 'wb') as f:
        pickle.dump(word2idx, f)
    print(f"Saved word2idx.pkl to {word2idx_path}")
    
    # 3. Create idx2word mapping
    idx2word = {idx: token for token, idx in word2idx.items()}
    
    # Save idx2word.pkl
    idx2word_path = os.path.join(output_dir, "idx2word.pkl")
    with open(idx2word_path, 'wb') as f:
        pickle.dump(idx2word, f)
    print(f"Saved idx2word.pkl to {idx2word_path}")
    
    # Print some statistics
    print("\n" + "="*50)
    print("Vocabulary Statistics:")
    print("="*50)
    print(f"Special tokens: {len(special_tokens)}")
    print(f"  {special_tokens}")
    print(f"\nUnique symbols: {len(all_labels)}")
    print(f"Total vocabulary size: {len(vocab)}")
    
    # Show label distribution
    label_counts = df['label'].value_counts()
    print(f"\nTop 10 most frequent symbols:")
    for label, count in label_counts.head(10).items():
        print(f"  {label}: {count} samples")
    
    print(f"\nTop 10 least frequent symbols:")
    for label, count in label_counts.tail(10).items():
        print(f"  {label}: {count} samples")
    
    # Verify the files were created
    print("\n" + "="*50)
    print("Files created:")
    print("="*50)
    print(f"✓ {vocab_txt_path}")
    print(f"✓ {word2idx_path}")
    print(f"✓ {idx2word_path}")
    
    return vocab, word2idx, idx2word

def verify_vocabulary(output_dir):
    """
    Verify the created vocabulary files
    """
    print("\n" + "="*50)
    print("Verifying vocabulary files...")
    print("="*50)
    
    # Load vocab.txt
    vocab_txt_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_txt_path, 'r', encoding='utf-8') as f:
        vocab_list = [line.strip() for line in f.readlines()]
    
    # Load word2idx
    word2idx_path = os.path.join(output_dir, "word2idx.pkl")
    with open(word2idx_path, 'rb') as f:
        word2idx = pickle.load(f)
    
    # Load idx2word
    idx2word_path = os.path.join(output_dir, "idx2word.pkl")
    with open(idx2word_path, 'rb') as f:
        idx2word = pickle.load(f)
    
    # Verify consistency
    assert len(vocab_list) == len(word2idx) == len(idx2word), "Vocabulary sizes don't match!"
    
    print(f"✓ All files have {len(vocab_list)} tokens")
    
    # Test some mappings
    print("\nTesting mappings:")
    test_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    if len(vocab_list) > 4:
        test_tokens.extend(vocab_list[4:min(9, len(vocab_list))])
    
    for token in test_tokens:
        idx = word2idx[token]
        recovered_token = idx2word[idx]
        print(f"  '{token}' -> {idx} -> '{recovered_token}' ✓")
    
    print("\n✓ Vocabulary files verified successfully!")
    
    return vocab_list, word2idx, idx2word

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create vocabulary files from symbols database')
    parser.add_argument('--csv', type=str, required=True, 
                        help='Path to symbols database CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for vocabulary files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify the created vocabulary files')
    
    args = parser.parse_args()
    
    # Create vocabulary
    vocab, word2idx, idx2word = create_vocabulary(args.csv, args.output)
    
    # Verify if requested
    if args.verify:
        verify_vocabulary(args.output)