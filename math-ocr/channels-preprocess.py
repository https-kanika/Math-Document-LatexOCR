# FAST MULTIPROCESSING PREPROCESSING with optimizations
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
import sys

def get_directional_kernels():
    k = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
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
    """Optimized directional map computation"""
    kernels = get_directional_kernels()
    edge_maps = []
    for kern in kernels:
        em = cv2.filter2D(image, cv2.CV_32F, kern)
        # Normalize and clip in one step
        em = np.clip(em / 255.0, 0, 1)
        edge_maps.append(em)
    return np.stack(edge_maps, axis=0)

def process_single_image(args):
    """Process a single image - used for multiprocessing"""
    idx, filename, label, dataset_root, split, output_dir = args
    
    try:
        # Construct paths
        img_path = os.path.join(dataset_root, split, filename)
        output_filename = os.path.splitext(filename)[0] + '.npy'  # Changed to .npy
        output_path = os.path.join(output_dir, split, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            return {
                'filename': output_filename,
                'normalized_label': label,
                'original_filename': filename
            }, None
        
        # Read and process image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, f"Could not read {img_path}"
        
        # Convert to float32 immediately
        image = image.astype(np.float32) / 255.0
        H, W = image.shape
        
        # Compute 9 channels - preallocate array
        channels = np.empty((9, H, W), dtype=np.float32)
        channels[0] = image
        channels[1:] = get_directional_maps(image)
        
        # Create subdirectories and save WITHOUT compression
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, channels)  # Changed from np.savez_compressed
        
        return {
            'filename': output_filename,
            'normalized_label': label,
            'original_filename': filename
        }, None
        
    except Exception as e:
        return None, f"Error processing {filename}: {str(e)}"

def precompute_9channel_dataset_parallel(csv_file, dataset_root, split, output_dir, num_workers=None, chunk_size=100):
    """
    Precompute dataset using multiprocessing - MUCH FASTER!
    """
    print(f"\n{'='*60}")
    print(f"PREPROCESSING {split.upper()} SPLIT (PARALLEL)")
    print(f"{'='*60}")
    sys.stdout.flush()  # Force flush output
    
    # Create output directory
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(csv_file)
    total_images = len(df)
    print(f"Found {total_images:,} images to process")
    print(f"CSV columns: {df.columns.tolist()}")
    sys.stdout.flush()
    
    # CHECK IF 'filename' COLUMN EXISTS
    if 'filename' not in df.columns:
        print(f"ERROR: 'filename' column not found in CSV!")
        print(f"Available columns: {df.columns.tolist()}")
        sys.stdout.flush()
        return 0, total_images
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores free
    num_workers = min(num_workers, cpu_count())  # Don't exceed available cores
    print(f"Using {num_workers} CPU cores")
    sys.stdout.flush()
    
    # Prepare arguments for multiprocessing
    args_list = [
        (idx, df.iloc[idx]['filename'], df.iloc[idx]['normalized_label'], 
         dataset_root, split, output_dir)
        for idx in range(total_images)
    ]
    
    # Process in parallel with progress bar
    processed_data = []
    failed_count = 0
    errors = []
    
    print(f"Processing in chunks of {chunk_size}...")
    print(f"Started at: {pd.Timestamp.now()}")
    sys.stdout.flush()
    
    import time
    start_time = time.time()
    last_update = start_time
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance
        results_iter = pool.imap_unordered(process_single_image, args_list, chunksize=chunk_size)
        
        # Manual progress tracking for Linux containers
        for i, (result, error) in enumerate(results_iter, 1):
            if result is not None:
                processed_data.append(result)
            else:
                failed_count += 1
                if error:
                    errors.append(error)
            
            # Print progress every 100 images or 5 seconds
            current_time = time.time()
            if i % 100 == 0 or (current_time - last_update) >= 5:
                elapsed = current_time - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_images - i) / rate if rate > 0 else 0
                
                print(f"Progress: {i}/{total_images} ({i/total_images*100:.1f}%) | "
                      f"Rate: {rate:.1f} img/s | "
                      f"ETA: {eta/60:.1f} min | "
                      f"Success: {len(processed_data)} | "
                      f"Failed: {failed_count}")
                sys.stdout.flush()
                last_update = current_time
    
    total_time = time.time() - start_time
    
    # Print errors
    if errors:
        print(f"\nFirst {min(10, len(errors))} errors:")
        for error in errors[:10]:
            print(f"  {error}")
        sys.stdout.flush()
    
    # Save mapping CSV
    mapping_df = pd.DataFrame(processed_data)
    mapping_csv = os.path.join(output_dir, f'{split}_mapping.csv')
    mapping_df.to_csv(mapping_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING {split.upper()} COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(processed_data):,} / {total_images:,}")
    print(f"Failed: {failed_count:,}")
    print(f"Success rate: {len(processed_data)/total_images*100:.2f}%")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average: {total_time/total_images*1000:.2f} ms/image")
    print(f"Output directory: {split_output_dir}")
    print(f"Mapping CSV: {mapping_csv}")
    
    # Calculate disk space used
    total_size_mb = 0
    for root, dirs, files in os.walk(split_output_dir):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total_size_mb += os.path.getsize(fp) / (1024 * 1024)
    print(f"Disk space used: {total_size_mb/1024:.2f} GB")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    return len(processed_data), failed_count

# ==============================
# RUN PREPROCESSING
# ==============================
if __name__ == '__main__':  # Important for Windows/Mac multiprocessing
    DATASET_ROOT = '/home/ie643_errorcode500/errorcode500-working/Mathwritting-100k'
    PROCESSED_ROOT = '/home/ie643_errorcode500/errorcode500-working/Mathwritting-100k-9ch'
    
    print("="*60)
    print("PARALLEL DATASET PREPROCESSING")
    print("="*60)
    print(f"Source: {DATASET_ROOT}")
    print(f"Output: {PROCESSED_ROOT}")
    print(f"Available CPUs: {cpu_count()}")
    sys.stdout.flush()
    
    # Determine optimal workers (use all but 2 cores)
    NUM_WORKERS = max(1, cpu_count() - 2)
    CHUNK_SIZE = 100  # Process 100 images per chunk
    
    print(f"Will use {NUM_WORKERS} workers")
    print(f"Chunk size: {CHUNK_SIZE}")
    print("="*60)
    sys.stdout.flush()
    
    input("Press Enter to start preprocessing...")
    
    import time
    start_time = time.time()
    
    total_success = 0
    total_failed = 0
    
    # Preprocess all splits
    for split in ['train', 'val', 'test']:
        csv_file = os.path.join(DATASET_ROOT, f'{split}_database.csv')
        if os.path.exists(csv_file):
            success, failed = precompute_9channel_dataset_parallel(
                csv_file, DATASET_ROOT, split, PROCESSED_ROOT, 
                num_workers=NUM_WORKERS,
                chunk_size=CHUNK_SIZE
            )
            total_success += success
            total_failed += failed
        else:
            print(f"Warning: {csv_file} not found, skipping {split} split")
            sys.stdout.flush()
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("ALL PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Total images processed: {total_success:,}")
    print(f"Total failed: {total_failed:,}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if total_success > 0:
        print(f"Average time per image: {total_time/total_success*1000:.2f} ms")
    print("\nNext steps:")
    print("1. Use Fast9ChDataset class for training")
    print("2. Point dataset to:", PROCESSED_ROOT)
    print("="*60)
    sys.stdout.flush()