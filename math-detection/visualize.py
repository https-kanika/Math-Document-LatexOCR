import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import cv2
import os
import pickle
from torch.utils.data import DataLoader
from model_final import FullyConvolutionalNetwork, GRUDecoder, reshape_fcn_output
from train import MathExpressionDataset, collate_fn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

class AttentionVisualizer:
    def __init__(self, encoder, decoder, device, word2idx_path, idx2word_path):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        
        with open(word2idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        with open(idx2word_path, 'rb') as f:
            self.idx2word = pickle.load(f)
        
        self.pad_idx = self.word2idx['<PAD>']
        self.start_idx = self.word2idx['<START>']
        self.end_idx = self.word2idx['<END>']
        
        self.encoder.eval()
        self.decoder.eval()
    
    def decode_with_attention(self, image, max_len=150, beam_width=1):
        """
        Decode image and collect attention weights at each step
        Returns: predicted_sequence, attention_weights_list, hidden_states_list
        """
        with torch.no_grad():
            # Encode image
            encoder_output = self.encoder(image)  # (1, D, H, W)
            annotations = reshape_fcn_output(encoder_output)  # (1, L, D)
            
            L = annotations.size(1)
            D = annotations.size(2)
            H = encoder_output.size(2)
            W = encoder_output.size(3)
            
            # Initialize
            h_t = torch.zeros(1, self.decoder.decoder_dim, device=self.device)
            c_t = torch.zeros(1, self.decoder.encoder_dim, device=self.device)
            beta_t = torch.zeros(1, L, device=self.device)
            
            predicted_sequence = [self.start_idx]
            attention_weights_list = []
            hidden_states_list = []
            context_vectors_list = []
            
            y_t = torch.tensor([self.start_idx], dtype=torch.long, device=self.device)
            
            for t in range(max_len):
                prob, h_t, c_t, alpha, beta_t = self.decoder.forward_step(
                    y_t, h_t, c_t, annotations, beta_t
                )
                
                # Store attention and hidden state
                attention_weights_list.append(alpha[0].cpu().numpy())  # (L,)
                hidden_states_list.append(h_t[0].cpu().numpy())  # (decoder_dim,)
                context_vectors_list.append(c_t[0].cpu().numpy())  # (encoder_dim,)
                
                # Get next token
                next_token = prob.argmax(dim=1).item()
                predicted_sequence.append(next_token)
                
                if next_token == self.end_idx:
                    break
                
                y_t = torch.tensor([next_token], dtype=torch.long, device=self.device)
            
            return {
                'predicted_sequence': predicted_sequence,
                'attention_weights': np.array(attention_weights_list),  # (seq_len, L)
                'hidden_states': np.array(hidden_states_list),  # (seq_len, decoder_dim)
                'context_vectors': np.array(context_vectors_list),  # (seq_len, encoder_dim)
                'encoder_output': encoder_output.cpu().numpy(),  # (1, D, H, W)
                'annotations': annotations.cpu().numpy(),  # (1, L, D)
                'feature_map_shape': (H, W)
            }
    
    def visualize_attention_on_image(self, image_path, output_dir, sample_name):
        """
        Visualize attention weights overlaid on original image
        """
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_h, original_w = image.shape
        
        # Prepare image tensor (5-channel)
        from train import make_5ch_from_image_gpu
        image_tensor = make_5ch_from_image_gpu(image_path, device=self.device)
        image_tensor = image_tensor.unsqueeze(0)  # (1, 5, H, W)
        
        # Decode with attention
        results = self.decode_with_attention(image_tensor)
        
        predicted_seq = results['predicted_sequence']
        attention_weights = results['attention_weights']  # (seq_len, L)
        H, W = results['feature_map_shape']
        
        # Convert predicted sequence to symbols
        pred_symbols = [self.idx2word.get(idx, f'<UNK:{idx}>') for idx in predicted_seq]
        
        # Reshape attention weights to spatial grid
        seq_len = attention_weights.shape[0]
        attention_spatial = attention_weights.reshape(seq_len, H, W)  # (seq_len, H, W)
        
        # Create visualization
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Heatmap grid visualization
        num_steps = min(seq_len, 20)  # Limit to first 20 tokens
        cols = 5
        rows = (num_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        for i in range(num_steps):
            ax = axes[i]
            
            # Resize attention map to original image size
            attn_map = attention_spatial[i]
            attn_resized = cv2.resize(attn_map, (original_w, original_h), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Overlay attention on image
            ax.imshow(image, cmap='gray', alpha=0.6)
            im = ax.imshow(attn_resized, cmap='jet', alpha=0.4)
            
            token_idx = predicted_seq[i+1] if i+1 < len(predicted_seq) else -1
            token_symbol = self.idx2word.get(token_idx, f'<UNK:{token_idx}>')
            
            ax.set_title(f'Step {i+1}: {token_symbol}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Hide unused subplots
        for i in range(num_steps, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Attention Visualization: {sample_name}\nPredicted: {" ".join(pred_symbols[1:-1])}', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_attention_grid.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Attention evolution heatmap (all timesteps)
        fig, ax = plt.subplots(figsize=(15, max(8, seq_len * 0.3)))
        
        # Flatten attention to (seq_len, H*W) and transpose for display
        attention_flat = attention_weights  # (seq_len, L)
        
        sns.heatmap(attention_flat, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
        
        # Set y-axis labels as predicted tokens
        token_labels = [self.idx2word.get(idx, f'<UNK:{idx}>') for idx in predicted_seq[1:]]
        ax.set_yticks(np.arange(len(token_labels)) + 0.5)
        ax.set_yticklabels(token_labels, fontsize=8)
        ax.set_xlabel('Spatial Location (flattened)', fontsize=10)
        ax.set_ylabel('Decoded Token', fontsize=10)
        ax.set_title(f'Attention Evolution Over Time: {sample_name}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_attention_evolution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Coverage tracking (beta values)
        if seq_len > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            cumulative_attention = np.cumsum(attention_weights, axis=0)  # (seq_len, L)
            
            # Plot cumulative attention for selected locations
            sample_locs = np.linspace(0, attention_weights.shape[1]-1, 10, dtype=int)
            
            for loc in sample_locs:
                ax.plot(cumulative_attention[:, loc], label=f'Loc {loc}', alpha=0.7)
            
            ax.set_xlabel('Decoding Step', fontsize=10)
            ax.set_ylabel('Cumulative Attention (Coverage)', fontsize=10)
            ax.set_title(f'Coverage Tracking: {sample_name}', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{sample_name}_coverage.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        return results
    
    def visualize_encoder_embeddings(self, dataloader, output_dir, num_samples=100):
        """
        Visualize encoder embeddings using PCA and t-SNE
        """
        all_embeddings = []
        all_labels = []
        all_filenames = []
        
        print(f"Collecting embeddings from {num_samples} samples...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                
                images = batch['images'].to(self.device)
                labels = batch['labels']
                filenames = batch['filenames']
                
                # Get encoder output
                encoder_output = self.encoder(images)  # (B, D, H, W)
                
                # Global average pooling to get fixed-size representation
                embedding = encoder_output.mean(dim=[2, 3])  # (B, D)
                
                all_embeddings.append(embedding.cpu().numpy())
                all_labels.extend(labels)
                all_filenames.extend(filenames)
        
        all_embeddings = np.vstack(all_embeddings)  # (num_samples, D)
        
        print(f"Collected {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # PCA visualization
        print("Computing PCA...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(all_embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                           alpha=0.6, s=50, c=np.arange(len(embeddings_pca)), cmap='viridis')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        ax.set_title('Encoder Embeddings - PCA Visualization', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'encoder_embeddings_pca.png'), dpi=150)
        plt.close()
        
        # t-SNE visualization
        print("Computing t-SNE (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_tsne = tsne.fit_transform(all_embeddings)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                           alpha=0.6, s=50, c=np.arange(len(embeddings_tsne)), cmap='viridis')
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('Encoder Embeddings - t-SNE Visualization', fontsize=14)
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'encoder_embeddings_tsne.png'), dpi=150)
        plt.close()
        
        # Save embedding data
        embedding_df = pd.DataFrame({
            'filename': all_filenames,
            'label': all_labels,
            'pca1': embeddings_pca[:, 0],
            'pca2': embeddings_pca[:, 1],
            'tsne1': embeddings_tsne[:, 0],
            'tsne2': embeddings_tsne[:, 1]
        })
        embedding_df.to_csv(os.path.join(output_dir, 'encoder_embeddings.csv'), index=False)
        
        print(f"Encoder embedding visualizations saved to {output_dir}")
        
        return embeddings_pca, embeddings_tsne, all_labels
    
    def visualize_encoder_feature_maps(self, image_path, output_dir, sample_name):
        """
        Visualize what the encoder is looking at by showing feature map activations
        """
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_h, original_w = image.shape
        
        # Prepare image tensor (5-channel)
        from train import make_5ch_from_image_gpu
        image_tensor = make_5ch_from_image_gpu(image_path, device=self.device)
        image_tensor = image_tensor.unsqueeze(0)  # (1, 5, H, W)
        
        # Get encoder output with feature maps from each block
        encoder_activations = self.get_encoder_activations(image_tensor)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize feature maps from each encoder block
        for block_name, activation in encoder_activations.items():
            self._plot_feature_maps(
                image, activation, block_name, 
                os.path.join(output_dir, f'{sample_name}_{block_name}_features.png')
            )
        
        # Visualize aggregated channel importance
        final_features = encoder_activations['block4']  # (1, 128, H, W)
        self._plot_channel_importance(
            image, final_features,
            os.path.join(output_dir, f'{sample_name}_channel_importance.png')
        )
        
        # Visualize spatial attention (average across channels)
        self._plot_spatial_activation(
            image, final_features,
            os.path.join(output_dir, f'{sample_name}_spatial_activation.png')
        )
        
        return encoder_activations
    
    def get_encoder_activations(self, image_tensor):
        """
        Hook into encoder to extract intermediate feature maps
        """
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for each block
        # Block 1
        self.encoder.maxpool1.register_forward_hook(hook_fn('block1'))
        # Block 2
        self.encoder.maxpool2.register_forward_hook(hook_fn('block2'))
        # Block 3
        self.encoder.maxpool3.register_forward_hook(hook_fn('block3'))
        # Block 4 (final)
        self.encoder.dropout4_4.register_forward_hook(hook_fn('block4'))
        
        with torch.no_grad():
            _ = self.encoder(image_tensor)
        
        return activations
    
    def _plot_feature_maps(self, original_image, activation, block_name, save_path):
        """
        Plot grid of feature maps from a specific encoder block
        """
        # activation shape: (1, C, H, W)
        activation = activation[0].cpu().numpy()  # (C, H, W)
        num_channels = activation.shape[0]
        
        # Select representative channels (max 16)
        num_display = min(16, num_channels)
        channel_importance = activation.reshape(num_channels, -1).max(axis=1)
        top_channels = np.argsort(channel_importance)[-num_display:]
        
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
        axes = axes.flatten()
        
        for idx, ch_idx in enumerate(top_channels):
            ax = axes[idx]
            feature_map = activation[ch_idx]  # (H, W)
            
            # Resize to original image size for overlay
            feature_resized = cv2.resize(
                feature_map, 
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize
            feature_resized = (feature_resized - feature_resized.min()) / (feature_resized.max() - feature_resized.min() + 1e-10)
            
            # Overlay on original image
            ax.imshow(original_image, cmap='gray', alpha=0.5)
            im = ax.imshow(feature_resized, cmap='hot', alpha=0.5)
            ax.set_title(f'Channel {ch_idx}\n(Importance: {channel_importance[ch_idx]:.3f})', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Hide unused subplots
        for idx in range(num_display, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{block_name} Feature Maps (Top {num_display} by activation)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_channel_importance(self, original_image, final_features, save_path):
        """
        Visualize which channels are most important (highest activation)
        """
        # final_features: (1, 128, H, W)
        features = final_features[0].cpu().numpy()  # (128, H, W)
        num_channels = features.shape[0]
        
        # Compute channel importance (max activation per channel)
        channel_importance = features.reshape(num_channels, -1).max(axis=1)
        
        # Sort channels by importance
        sorted_indices = np.argsort(channel_importance)[::-1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Channel importance bar chart
        ax1.bar(range(num_channels), channel_importance[sorted_indices], alpha=0.7)
        ax1.set_xlabel('Channel (sorted by importance)', fontsize=12)
        ax1.set_ylabel('Max Activation', fontsize=12)
        ax1.set_title('Channel Importance Distribution', fontsize=14)
        ax1.grid(alpha=0.3)
        
        # Plot 2: Weighted average of top-k channels overlaid on image
        top_k = 32
        top_channels = sorted_indices[:top_k]
        
        weighted_map = np.zeros((features.shape[1], features.shape[2]))
        for ch_idx in top_channels:
            weighted_map += features[ch_idx] * channel_importance[ch_idx]
        
        # Resize to original image size
        weighted_map_resized = cv2.resize(
            weighted_map,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        weighted_map_resized = (weighted_map_resized - weighted_map_resized.min()) / (weighted_map_resized.max() - weighted_map_resized.min() + 1e-10)
        
        ax2.imshow(original_image, cmap='gray', alpha=0.5)
        im = ax2.imshow(weighted_map_resized, cmap='jet', alpha=0.5)
        ax2.set_title(f'Weighted Activation Map (Top {top_k} Channels)', fontsize=14)
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_spatial_activation(self, original_image, final_features, save_path):
        """
        Visualize spatial activation pattern (average across all channels)
        """
        # final_features: (1, 128, H, W)
        features = final_features[0].cpu().numpy()  # (128, H, W)
        
        # Compute different aggregation strategies
        avg_activation = features.mean(axis=0)  # Average across channels
        max_activation = features.max(axis=0)   # Max across channels
        std_activation = features.std(axis=0)   # Std across channels
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for ax, activation, title in zip(
            axes,
            [avg_activation, max_activation, std_activation],
            ['Mean Activation', 'Max Activation', 'Std Activation']
        ):
            # Resize to original image size
            activation_resized = cv2.resize(
                activation,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Normalize
            activation_resized = (activation_resized - activation_resized.min()) / (activation_resized.max() - activation_resized.min() + 1e-10)
            
            ax.imshow(original_image, cmap='gray', alpha=0.5)
            im = ax.imshow(activation_resized, cmap='plasma', alpha=0.5)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle('Spatial Activation Patterns (Aggregated Across Channels)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_grad_cam(self, image_path, output_dir, sample_name, target_layer='dropout4_4'):
        """
        Grad-CAM visualization to see what encoder focuses on for specific predictions
        """
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_h, original_w = image.shape
        
        from train import make_5ch_from_image_gpu
        image_tensor = make_5ch_from_image_gpu(image_path, device=self.device)
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)  # (1, 5, H, W)
        
        # Forward pass
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks on target layer
        target_module = getattr(self.encoder, target_layer)
        fh = target_module.register_forward_hook(forward_hook)
        bh = target_module.register_full_backward_hook(backward_hook)
        
        # Forward pass
        encoder_output = self.encoder(image_tensor)  # (1, 128, H, W)
        
        # Compute gradients w.r.t. encoder output mean (global feature importance)
        encoder_output.mean().backward()
        
        # Get activation and gradient
        activation = activations[0].detach()  # (1, 128, H, W)
        gradient = gradients[0].detach()      # (1, 128, H, W)
        
        # Compute channel weights (global average pooling of gradients)
        weights = gradient.mean(dim=[2, 3], keepdim=True)  # (1, 128, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activation).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)  # ReLU to keep only positive contributions
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        
        # Resize to original image size
        cam_resized = cv2.resize(cam, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Visualize
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, cmap='gray', alpha=0.5)
        axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Grad-CAM Overlay', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f'Grad-CAM Visualization: {sample_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{sample_name}_gradcam.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Remove hooks
        fh.remove()
        bh.remove()
        
        return cam_resized
    
    

    

    
    def analyze_attention_patterns(self, dataloader, output_dir, num_samples=50):
        """
        Analyze common attention patterns and failure modes
        """
        attention_stats = {
            'max_attention': [],
            'mean_attention': [],
            'std_attention': [],
            'entropy': [],
            'coverage_max': [],
            'sequence_length': [],
            'wer': [],
            'labels': [],
            'predictions': []
        }
        
        print(f"Analyzing attention patterns for {num_samples} samples...")
        
        from train import compute_wer_detailed
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            labels = batch['labels']
            
            for i in range(images.size(0)):
                image = images[i:i+1]
                target = targets[i].cpu().numpy().tolist()
                
                # Decode with attention
                results = self.decode_with_attention(image)
                
                pred_seq = results['predicted_sequence']
                attention_weights = results['attention_weights']
                
                # Compute statistics
                attention_stats['max_attention'].append(attention_weights.max(axis=1).mean())
                attention_stats['mean_attention'].append(attention_weights.mean())
                attention_stats['std_attention'].append(attention_weights.std())
                
                # Compute entropy (measure of attention spread)
                entropy = -(attention_weights * np.log(attention_weights + 1e-10)).sum(axis=1).mean()
                attention_stats['entropy'].append(entropy)
                
                # Coverage
                cumulative = np.cumsum(attention_weights, axis=0)
                attention_stats['coverage_max'].append(cumulative[-1].max())
                
                attention_stats['sequence_length'].append(len(pred_seq))
                
                # Compute WER
                ref = [t for t in target[1:] if t not in [self.pad_idx, self.end_idx]]
                hyp = [t for t in pred_seq[1:] if t not in [self.pad_idx, self.end_idx]]
                if hyp and hyp[-1] == self.end_idx:
                    hyp = hyp[:-1]
                
                wer_result = compute_wer_detailed(ref, hyp)
                attention_stats['wer'].append(wer_result['wer'])
                
                attention_stats['labels'].append(labels[i])
                pred_symbols = [self.idx2word.get(idx, f'<UNK:{idx}>') for idx in pred_seq[1:-1]]
                attention_stats['predictions'].append(' '.join(pred_symbols))
        
        # Create analysis plots
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: WER vs Max Attention
        axes[0, 0].scatter(attention_stats['max_attention'], attention_stats['wer'], alpha=0.6)
        axes[0, 0].set_xlabel('Mean Max Attention per Step')
        axes[0, 0].set_ylabel('WER')
        axes[0, 0].set_title('WER vs Attention Peak')
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: WER vs Attention Entropy
        axes[0, 1].scatter(attention_stats['entropy'], attention_stats['wer'], alpha=0.6)
        axes[0, 1].set_xlabel('Mean Attention Entropy')
        axes[0, 1].set_ylabel('WER')
        axes[0, 1].set_title('WER vs Attention Spread')
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: WER vs Coverage
        axes[0, 2].scatter(attention_stats['coverage_max'], attention_stats['wer'], alpha=0.6)
        axes[0, 2].set_xlabel('Max Coverage Value')
        axes[0, 2].set_ylabel('WER')
        axes[0, 2].set_title('WER vs Coverage')
        axes[0, 2].grid(alpha=0.3)
        
        # Plot 4: Attention statistics histogram
        axes[1, 0].hist(attention_stats['max_attention'], bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Mean Max Attention')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Max Attention')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 5: Entropy distribution
        axes[1, 1].hist(attention_stats['entropy'], bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Mean Entropy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Attention Entropy')
        axes[1, 1].grid(alpha=0.3)
        
        # Plot 6: Sequence length vs WER
        axes[1, 2].scatter(attention_stats['sequence_length'], attention_stats['wer'], alpha=0.6)
        axes[1, 2].set_xlabel('Predicted Sequence Length')
        axes[1, 2].set_ylabel('WER')
        axes[1, 2].set_title('WER vs Sequence Length')
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_pattern_analysis.png'), dpi=150)
        plt.close()
        
        # Save statistics
        stats_df = pd.DataFrame(attention_stats)
        stats_df.to_csv(os.path.join(output_dir, 'attention_statistics.csv'), index=False)
        
        print(f"Attention pattern analysis saved to {output_dir}")
        
        return attention_stats


def main():
    """Main visualization script"""
    # Configuration
    DATA_DIR = r"C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessedCrome2014Data"
    TEST_CSV = r"C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessedCrome2014Data\test_database_cleaned.csv"
    WORD2IDX_PATH = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\pipeline\vocab\word2idx.pkl"
    IDX2WORD_PATH = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\pipeline\vocab\idx2word.pkl"
    BASE_IMAGE_DIR = r"C:\Users\kani1\Desktop\IE643\custom-dataset\ProccessedCrome2014Data\test"
    CHECKPOINT_PATH = r"C:\Users\kani1\Desktop\IE643\Math-Document-LatexOCR\math-detection\checkpoint_epoch_45.pth"
    OUTPUT_DIR = 'visualizations'
    
    # Model parameters  
    with open(WORD2IDX_PATH, 'rb') as f:
        word2idx = pickle.load(f)
    VOCAB_SIZE = len(set(word2idx.values()))
    
    EMBEDDING_DIM = 256
    DECODER_DIM = 256
    ENCODER_DIM = 128
    ATTENTION_DIM = 512
    COVERAGE_KERNEL_SIZE = 11
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    encoder = FullyConvolutionalNetwork()
    decoder = GRUDecoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        decoder_dim=DECODER_DIM,
        encoder_dim=ENCODER_DIM,
        attention_dim=ATTENTION_DIM,
        kernel_size=COVERAGE_KERNEL_SIZE
    )
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    print("Models loaded successfully!")
    
    # Create visualizer
    visualizer = AttentionVisualizer(
        encoder=encoder,
        decoder=decoder,
        device=device,
        word2idx_path=WORD2IDX_PATH,
        idx2word_path=IDX2WORD_PATH
    )
    
    # Create test dataset
    test_dataset = MathExpressionDataset(
        csv_path=TEST_CSV,
        word2idx_path=WORD2IDX_PATH,
        base_image_dir=BASE_IMAGE_DIR,
        device=device,
        subset_size=100  # Limit for visualization
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print("\n" + "="*80)
    print("VISUALIZING ATTENTION FOR SAMPLE IMAGES")
    print("="*80)
    
    attention_output_dir = os.path.join(OUTPUT_DIR, 'attention_maps')
    os.makedirs(attention_output_dir, exist_ok=True)
    
    # ADD: Encoder feature visualization directory
    encoder_features_dir = os.path.join(OUTPUT_DIR, 'encoder_features')
    os.makedirs(encoder_features_dir, exist_ok=True)
    
    gradcam_dir = os.path.join(OUTPUT_DIR, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Visualize first 10 samples
    for idx in range(min(10, len(test_dataset))):
        sample = test_dataset[idx]
        image_path = os.path.join(BASE_IMAGE_DIR, sample['filename'])
        sample_name = os.path.splitext(sample['filename'])[0]
        
        print(f"Processing sample {idx+1}/10: {sample['filename']}")
        print(f"Ground truth: {sample['label']}")
        
        # Attention visualization
        results = visualizer.visualize_attention_on_image(
            image_path=image_path,
            output_dir=attention_output_dir,
            sample_name=sample_name
        )
        
        pred_symbols = [visualizer.idx2word.get(i, f'<UNK:{i}>') 
                       for i in results['predicted_sequence'][1:-1]]
        print(f"Prediction: {' '.join(pred_symbols)}")
        
        # ADD: Encoder feature visualization
        print(f"  Visualizing encoder features...")
        visualizer.visualize_encoder_feature_maps(
            image_path=image_path,
            output_dir=encoder_features_dir,
            sample_name=sample_name
        )
        
        # ADD: Grad-CAM visualization
        print(f"  Generating Grad-CAM...")
        visualizer.visualize_grad_cam(
            image_path=image_path,
            output_dir=gradcam_dir,
            sample_name=sample_name
        )
        
        print()
    
    # 2. Visualize encoder embeddings
    print("\n" + "="*80)
    print("VISUALIZING ENCODER EMBEDDINGS")
    print("="*80)
    
    embedding_output_dir = os.path.join(OUTPUT_DIR, 'embeddings')
    visualizer.visualize_encoder_embeddings(
        dataloader=test_loader,
        output_dir=embedding_output_dir,
        num_samples=100
    )
    
    # 3. Analyze attention patterns
    print("\n" + "="*80)
    print("ANALYZING ATTENTION PATTERNS")
    print("="*80)
    
    analysis_output_dir = os.path.join(OUTPUT_DIR, 'analysis')
    attention_stats = visualizer.analyze_attention_patterns(
        dataloader=test_loader,
        output_dir=analysis_output_dir,
        num_samples=50
    )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - Attention maps: {attention_output_dir}/")
    print(f"  - Encoder features: {encoder_features_dir}/")  # ADD
    print(f"  - Grad-CAM: {gradcam_dir}/")  # ADD
    print(f"  - Encoder embeddings: {embedding_output_dir}/")
    print(f"  - Pattern analysis: {analysis_output_dir}/")



if __name__ == '__main__':
    main()