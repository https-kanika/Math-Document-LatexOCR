import re
from collections import Counter
import csv

def clean_latex_label(latex_string):
    """
    Remove leading and trailing $ signs from LaTeX labels.
    Also removes style commands: \mbox{...}, \hbox{...}, \mathrm{...}, \vtop{...}
    Also removes delimiter sizing: \Big, \Bigg, \left, \right, \limits
    """
    # Strip whitespace first
    cleaned = latex_string.strip()
    
    # Remove leading $
    if cleaned.startswith('$'):
        cleaned = cleaned[1:]
    
    # Remove trailing $
    if cleaned.endswith('$'):
        cleaned = cleaned[:-1]
    
    # ✅ Remove style commands with braces: \mbox, \hbox, \mathrm, \vtop
    style_commands = [r'\mbox', r'\hbox', r'\mathrm', r'\vtop']
    
    for cmd in style_commands:
        while cmd in cleaned:
            # Find command followed by optional spaces and {
            pattern = re.escape(cmd) + r'\s*\{'
            match = re.search(pattern, cleaned)
            if not match:
                break
            
            start = match.start()
            brace_start = match.end() - 1  # Position of opening {
            
            # Find matching closing brace
            brace_count = 1
            i = brace_start + 1
            while i < len(cleaned) and brace_count > 0:
                if cleaned[i] == '{':
                    brace_count += 1
                elif cleaned[i] == '}':
                    brace_count -= 1
                i += 1
            
            if brace_count == 0:
                # Extract content inside braces
                content = cleaned[brace_start + 1:i - 1]
                # Replace \cmd{content} with just {content}
                cleaned = cleaned[:start] + '{' + content + '}' + cleaned[i:]
            else:
                # Malformed - just remove the command
                cleaned = cleaned[:start] + cleaned[match.end():]
    
    # ✅ Remove delimiter sizing commands (no braces, just delete them)
    # Order matters: remove longer commands first (\Bigg before \Big)
    delimiter_commands = [
        r'\\Bigg\s*',   # \Bigg
        r'\\bigg\s*',   # \bigg
        r'\\Big\s*',    # \Big
        r'\\big\s*',    # \big
        r'\\left\s*',   # \left
        r'\\right\s*',  # \right
        r'\\limits\s*'  # \limits
    ]
    
    for pattern in delimiter_commands:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Strip any remaining whitespace
    return cleaned.strip()


def tokenize_latex(latex_string):
    """
    Tokenize LaTeX string so that commands like \frac, \sqrt are single tokens.
    Also handles symbols, numbers, letters, and special characters.
    """
    tokens = []
    i = 0
    
    while i < len(latex_string):
        # Check for LaTeX commands (start with backslash)
        if latex_string[i] == '\\':
            # Find the end of the command
            j = i + 1
            # Command names are alphabetic
            while j < len(latex_string) and latex_string[j].isalpha():
                j += 1
            
            # If we found a command
            if j > i + 1:
                tokens.append(latex_string[i:j])
                i = j
            else:
                # Special case: backslash followed by non-alpha (like \{, \}, \|)
                if j < len(latex_string):
                    tokens.append(latex_string[i:j+1])
                    i = j + 1
                else:
                    tokens.append(latex_string[i])
                    i += 1
        
        # Skip whitespace
        elif latex_string[i].isspace():
            i += 1
        
        # Handle brackets, braces, and other special characters as single tokens
        elif latex_string[i] in '{}[]()^_=+-*/|<>!.,:;':
            tokens.append(latex_string[i])
            i += 1
        
        # Handle multi-character operators or symbols
        else:
            # Single character token (digit, letter, punctuation)
            tokens.append(latex_string[i])
            i += 1
    
    return tokens


def process_csv_and_clean(input_csv, output_csv):
    """
    Read input CSV, clean labels by removing $ signs, and save to new CSV.
    Filters out examples containing problematic tokens.
    Returns list of cleaned labels.
    """
    # ✅ Define tokens to filter out
    problematic_tokens = [
        r'\exists',
        r'\gtM', r'\gtp', 
        r'\ltq', r'\ltl', r'\ltN',
        r'\dots',  # Will normalize to \ldots instead
        r'\lbrack', r'\rbrack'
    ]
    
    cleaned_data = []
    filtered_count = 0
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            original_label = row['label']
            
            # ✅ Check if label contains any problematic tokens
            should_filter = False
            for token in problematic_tokens:
                if token in original_label:
                    should_filter = True
                    filtered_count += 1
                    break
            
            # Skip this row if it contains problematic tokens
            if should_filter:
                continue
            
            # Clean the label
            cleaned_label = clean_latex_label(original_label)
            
            # Update the row
            row['label'] = cleaned_label
            cleaned_data.append(row)
    
    # Write cleaned data to new CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_data)
    
    print(f"Cleaned CSV saved to: {output_csv}")
    print(f"Total rows processed: {len(cleaned_data)}")
    print(f"Rows filtered out (containing problematic tokens): {filtered_count}")
    
    return cleaned_data


def build_vocabulary(cleaned_data):
    """
    Build vocabulary from cleaned label data.
    Returns a dictionary mapping tokens to indices and vice versa.
    Adds special tokens: <PAD>, <START>, <END>
    """
    token_counter = Counter()
    
    # Tokenize all labels
    for row in cleaned_data:
        latex_label = row['label']
        tokens = tokenize_latex(latex_label)
        token_counter.update(tokens)
    
    # Sort tokens by frequency (most common first)
    sorted_tokens = [token for token, count in token_counter.most_common()]
    
    # Add special tokens at the beginning
    special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
    all_tokens = special_tokens + sorted_tokens
    
    # Create vocabulary mappings
    token2idx = {token: idx for idx, token in enumerate(all_tokens)}
    idx2token = {idx: token for token, idx in token2idx.items()}
    
    # Add special tokens to counter with count 0 for tracking
    for special_token in special_tokens:
        token_counter[special_token] = 0
    
    return token2idx, idx2token, token_counter


import re
from collections import Counter
import csv
import pickle


def save_vocabulary(token2idx, token_counter, output_file='vocab.txt'):
    """
    Save vocabulary to a text file with token frequencies.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total unique tokens: {len(token2idx)}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Index\tToken\tFrequency\n")
        f.write("-" * 50 + "\n")
        
        for token, idx in sorted(token2idx.items(), key=lambda x: x[1]):
            count = token_counter[token]
            # Escape special characters for display
            display_token = repr(token) if token in ['\t', '\n', ' '] else token
            f.write(f"{idx}\t{display_token}\t{count}\n")
    
    print(f"\nVocabulary saved to: {output_file}")
    print(f"Total unique tokens: {len(token2idx)}")


def save_pickle_files(token2idx, idx2token, word2idx_file='word2idx.pkl', idx2word_file='idx2word.pkl'):
    """
    Save token2idx and idx2token dictionaries as pickle files.
    """
    # Save word2idx (token2idx)
    with open(word2idx_file, 'wb') as f:
        pickle.dump(token2idx, f)
    print(f"word2idx saved to: {word2idx_file}")
    
    # Save idx2word (idx2token)
    with open(idx2word_file, 'wb') as f:
        pickle.dump(idx2token, f)
    print(f"idx2word saved to: {idx2word_file}")


# Example usage
if __name__ == "__main__":
    input_csv = r"C:\Users\kani1\Desktop\IE643\custom-dataset\ProcessedMathWrittingSymbols\symbols_database.csv"  # Your original CSV file
    output_csv = "symbols_database_cleaned.csv"  # Cleaned CSV output

    print("Step 1: Cleaning CSV and removing $ signs...")
    cleaned_data = process_csv_and_clean(input_csv, output_csv)
    
    print("\nStep 2: Building vocabulary from cleaned labels...")
    token2idx, idx2token, token_counter = build_vocabulary(cleaned_data)
    
    print("\nStep 3: Saving vocabulary...")
    save_vocabulary(token2idx, token_counter)
    
    print("\nStep 4: Saving pickle files...")
    save_pickle_files(token2idx, idx2token)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)
    print(f"Total unique tokens: {len(token2idx)}")
    print(f"\nTop 20 most common tokens:")
    for token, count in token_counter.most_common(20):
        display_token = repr(token) if len(token) == 1 and not token.isalnum() else token
        print(f"  {display_token}: {count} times")
    
    # Test tokenization with cleaned labels
    print("\n" + "=" * 50)
    print("TEST TOKENIZATION")
    print("=" * 50)
    test_cases = [
    r"$\frac{1}{2}$",
    r"$\sqrt{x}$",
    r"$x^{2}$",
    r"$\theta_{i}$",
    r"$\mbox{text}$",
    r"$\mbox { hello } + \mbox { world }$",
    r"$\left(\frac{1}{2}\right)$",  # ✅ Test \left \right removal
    r"$\Bigg(\sum_{i=1}^{n}\Bigg)$",  # ✅ Test \Bigg removal
    r"$\Big[\frac{a}{b}\Big]$",  # ✅ Test \Big removal
    r"$\sum\limits_{i=1}^{n} x_i$",  # ✅ Test \limits removal
    r" \sum _ { { \mbox { Y } = M } } ^ { { \mbox { P } } ^ { \mbox { r } } \left ( \mbox { x } \right ) } { { G + V } } "
]
    for test in test_cases:
        cleaned = clean_latex_label(test)
        tokens = tokenize_latex(cleaned)
        print(f"Original: {test}")
        print(f"Cleaned:  {cleaned}")
        print(f"Tokens:   {tokens}")
        print(f"Token IDs: {[token2idx.get(token, -1) for token in tokens]}")

        