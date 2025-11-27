import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm

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

    dirs = torch.sqrt(dirs / (dirs.amax(dim=(2, 3), keepdim=True) + 1e-12))

    # Stack grayscale + directional: (1, 1, H, W) + (1, 4, H, W) = (1, 5, H, W)
    five = torch.cat([gray, dirs], dim=1)
    
    return five.squeeze(0)  # (5, H, W)


class FiveChannelVisualizer:
    """
    Visualizer for 5-channel feature extraction pipeline
    """
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.channel_names = ['Grayscale', '0° (Horizontal)', '45° (Diagonal ↗)', 
                             '90° (Vertical)', '135° (Diagonal ↖)']
        self.direction_angles = [0, 45, 90, 135]
        
    def visualize_single_image(self, image_path, output_dir, sample_name):
        """
        Visualize all 5 channels for a single image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load original image
        original_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Generate 5-channel features
        features_5ch = make_5ch_from_image_gpu(
            image_path, 
            blur_sigma=1.0, 
            thick_radius=1, 
            device=self.device
        )  # (5, H, W)
        
        features_np = features_5ch.cpu().numpy()
        
        # ===== VISUALIZATION 1: Grid of all 5 channels =====
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 5 feature channels
        for i in range(5):
            ax = axes[i + 1]
            channel = features_np[i]
            
            # Choose colormap based on channel
            if i == 0:  # Grayscale
                cmap = 'gray'
            else:  # Directional channels
                cmap = 'hot'
            
            im = ax.imshow(channel, cmap=cmap)
            ax.set_title(self.channel_names[i], fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle(f'5-Channel Feature Extraction: {sample_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_5channels_grid.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # ===== VISUALIZATION 2: Overlay on original image =====
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        for i in range(5):
            channel = features_np[i]
            
            # Normalize for better visualization
            channel_norm = (channel - channel.min()) / (channel.max() - channel.min() + 1e-10)
            
            axes[i].imshow(original_img, cmap='gray', alpha=0.5)
            axes[i].imshow(channel_norm, cmap='jet', alpha=0.5)
            axes[i].set_title(self.channel_names[i], fontsize=11, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle(f'5-Channel Overlay: {sample_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_5channels_overlay.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # ===== VISUALIZATION 3: Statistical Analysis =====
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Channel intensity distributions
        axes[0, 0].set_title('Channel Intensity Distributions', fontsize=12, fontweight='bold')
        for i in range(5):
            channel_flat = features_np[i].flatten()
            axes[0, 0].hist(channel_flat, bins=50, alpha=0.6, label=self.channel_names[i])
        axes[0, 0].set_xlabel('Intensity Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)
        
        # Channel activation statistics (bar plot)
        channel_stats = {
            'Mean': [features_np[i].mean() for i in range(5)],
            'Std': [features_np[i].std() for i in range(5)],
            'Max': [features_np[i].max() for i in range(5)],
            'Non-zero %': [(features_np[i] > 0.01).sum() / features_np[i].size * 100 for i in range(5)]
        }
        
        x = np.arange(5)
        width = 0.2
        
        axes[0, 1].bar(x - 1.5*width, channel_stats['Mean'], width, label='Mean', alpha=0.8)
        axes[0, 1].bar(x - 0.5*width, channel_stats['Std'], width, label='Std', alpha=0.8)
        axes[0, 1].bar(x + 0.5*width, channel_stats['Max'], width, label='Max', alpha=0.8)
        axes[0, 1].set_xlabel('Channel')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].set_title('Channel Statistics', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(['Gray', '0°', '45°', '90°', '135°'], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Spatial activation map (average across directional channels)
        directional_avg = features_np[1:].mean(axis=0)  # Average of 4 directional channels
        
        axes[1, 0].imshow(original_img, cmap='gray', alpha=0.5)
        im = axes[1, 0].imshow(directional_avg, cmap='plasma', alpha=0.5)
        axes[1, 0].set_title('Average Directional Activation', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        # Channel correlation heatmap
        # Flatten each channel and compute correlation
        channels_flat = np.array([features_np[i].flatten() for i in range(5)])
        correlation_matrix = np.corrcoef(channels_flat)
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                   xticklabels=['Gray', '0°', '45°', '90°', '135°'],
                   yticklabels=['Gray', '0°', '45°', '90°', '135°'],
                   cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Channel Correlation Matrix', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_5channels_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # ===== VISUALIZATION 4: Edge strength and orientation =====
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Combine directional channels to show dominant orientation
        directional_channels = features_np[1:]  # (4, H, W)
        
        # Max directional response
        max_response = directional_channels.max(axis=0)
        dominant_direction = directional_channels.argmax(axis=0)
        
        # Create color-coded orientation map
        orientation_map = np.zeros((*dominant_direction.shape, 3))
        colors = [
            [1, 0, 0],      # Red for 0° (horizontal)
            [0, 1, 0],      # Green for 45°
            [0, 0, 1],      # Blue for 90° (vertical)
            [1, 1, 0]       # Yellow for 135°
        ]
        
        for i in range(4):
            mask = dominant_direction == i
            for c in range(3):
                orientation_map[:, :, c] += mask * colors[i][c] * max_response
        
        # Normalize
        orientation_map = np.clip(orientation_map, 0, 1)
        
        axes[0].imshow(max_response, cmap='hot')
        axes[0].set_title('Maximum Directional Response', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(orientation_map)
        axes[1].set_title('Dominant Orientation Map\n(Red=0°, Green=45°, Blue=90°, Yellow=135°)', 
                         fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(original_img, cmap='gray', alpha=0.5)
        axes[2].imshow(orientation_map, alpha=0.5)
        axes[2].set_title('Orientation Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_orientation_map.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved for {sample_name}")
        
        return features_5ch
    
    def compare_preprocessing_params(self, image_path, output_dir, sample_name):
        """
        Compare different preprocessing parameters
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Test different blur_sigma values
        blur_sigmas = [0.0, 0.5, 1.0, 2.0]
        thick_radii = [0, 1, 2]
        
        fig, axes = plt.subplots(len(thick_radii), len(blur_sigmas), 
                                figsize=(20, 12))
        
        for i, thick_radius in enumerate(thick_radii):
            for j, blur_sigma in enumerate(blur_sigmas):
                features = make_5ch_from_image_gpu(
                    image_path, 
                    blur_sigma=blur_sigma, 
                    thick_radius=thick_radius,
                    device=self.device
                )
                
                # Average directional channels for visualization
                directional_avg = features[1:].mean(dim=0).cpu().numpy()
                
                axes[i, j].imshow(directional_avg, cmap='hot')
                axes[i, j].set_title(f'Blur={blur_sigma}, Thick={thick_radius}', fontsize=10)
                axes[i, j].axis('off')
        
        plt.suptitle(f'Preprocessing Parameter Comparison: {sample_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_param_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Parameter comparison saved for {sample_name}")
    
    def batch_visualize(self, image_dir, output_dir, num_samples=10, pattern='*.png'):
        """
        Visualize multiple images from a directory
        """
        image_paths = list(Path(image_dir).glob(pattern))
        
        if len(image_paths) == 0:
            print(f"No images found in {image_dir} with pattern {pattern}")
            return
        
        # Limit to num_samples
        image_paths = image_paths[:num_samples]
        
        print(f"\nVisualizing {len(image_paths)} images...")
        
        for img_path in tqdm(image_paths):
            sample_name = img_path.stem
            
            try:
                self.visualize_single_image(
                    str(img_path), 
                    output_dir, 
                    sample_name
                )
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    """
    Main visualization script
    """
    # Configuration
    IMAGE_DIR = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\math-detection\tmp"
    OUTPUT_DIR = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\math-detection\5channel_visualizations"
    NUM_SAMPLES = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("5-CHANNEL FEATURE VISUALIZATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Input directory: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print("="*80 + "\n")
    
    # Create visualizer
    visualizer = FiveChannelVisualizer(device=DEVICE)
    
    # Batch visualization
    visualizer.batch_visualize(
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        pattern='*.png'
    )
    
    # Optional: Visualize specific image with parameter comparison
    specific_images = list(Path(IMAGE_DIR).glob('*.png'))[:3]
    
    if specific_images:
        print("\n" + "="*80)
        print("PARAMETER COMPARISON FOR SAMPLE IMAGES")
        print("="*80 + "\n")
        
        param_comparison_dir = os.path.join(OUTPUT_DIR, 'parameter_comparison')
        
        for img_path in specific_images:
            sample_name = img_path.stem
            print(f"Comparing parameters for: {sample_name}")
            visualizer.compare_preprocessing_params(
                str(img_path),
                param_comparison_dir,
                sample_name
            )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - Individual visualizations: {OUTPUT_DIR}/")
    print(f"  - Parameter comparisons: {OUTPUT_DIR}/parameter_comparison/")


if __name__ == '__main__':
    main()