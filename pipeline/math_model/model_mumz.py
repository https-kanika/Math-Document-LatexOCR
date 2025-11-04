import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(FullyConvolutionalNetwork, self).__init__()
        # First block: 4 conv layers (5 -> 32 channels)
        self.conv1_1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(32)
        self.conv1_4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1_4 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second block: 4 conv layers (32 -> 64 channels)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Third block: 4 conv layers (64 -> 64 channels)
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(64)
        self.conv3_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_4 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fourth block: 4 conv layers with dropout (64 -> 128 channels)
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.dropout4_1 = nn.Dropout2d(p=0.2)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.dropout4_2 = nn.Dropout2d(p=0.2)
        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(128)
        self.dropout4_3 = nn.Dropout2d(p=0.2)
        self.conv4_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_4 = nn.BatchNorm2d(128)
        self.dropout4_4 = nn.Dropout2d(p=0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First block
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.relu(self.bn1_3(self.conv1_3(x)))
        x = self.relu(self.bn1_4(self.conv1_4(x)))
        x = self.maxpool1(x)
        # Second block
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.relu(self.bn2_3(self.conv2_3(x)))
        x = self.relu(self.bn2_4(self.conv2_4(x)))
        x = self.maxpool2(x)
        # Third block
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.relu(self.bn3_4(self.conv3_4(x)))
        x = self.maxpool3(x)
        # Fourth block with dropout
        x = self.dropout4_1(self.relu(self.bn4_1(self.conv4_1(x))))
        x = self.dropout4_2(self.relu(self.bn4_2(self.conv4_2(x))))
        x = self.dropout4_3(self.relu(self.bn4_3(self.conv4_3(x))))
        x = self.dropout4_4(self.relu(self.bn4_4(self.conv4_4(x))))
        return x

def reshape_fcn_output(fcn_output):
    """
    Reshape FCN output from (B, D, H, W) to variable-length grid format (B, L, D)
    where L = H × W
    Args:
    fcn_output: torch.Tensor of shape (B, D, H, W) where
        B = batch size
        D = number of channels/feature dimensions
        H = height
        W = width
    Returns:
    reshaped: torch.Tensor of shape (B, L, D) where L = H × W
    Each element a_i is a D-dimensional annotation
    """
    B, D, H, W = fcn_output.shape
    L = H * W
    # Permute to (B, H, W, D) then reshape to (B, L, D)
    reshaped = fcn_output.permute(0, 2, 3, 1).reshape(B, L, D)
    return reshaped

def reshape_fcn_output_single(fcn_output):
    """
    Reshape single FCN output from (D, H, W) to variable-length grid format (L, D)
    where L = H × W
    Args:
    fcn_output: torch.Tensor of shape (D, H, W) where
        D = number of channels/feature dimensions
        H = height
        W = width
    Returns:
    reshaped: torch.Tensor of shape (L, D) where L = H × W
    a = {a1, ..., aL}, ai ∈ R^D
    """
    D, H, W = fcn_output.shape
    L = H * W
    # Permute to (H, W, D) then reshape to (L, D)
    reshaped = fcn_output.permute(1, 2, 0).reshape(L, D)
    return reshaped

class AttentionWithCoverage(nn.Module):
    """
    Attention mechanism with coverage vector to prevent over-parsing and under-parsing.
    """
    def __init__(self, attention_dim, encoder_dim, decoder_dim, kernel_size=11):
        """
        Args:
        attention_dim (n'): Dimension of attention network
        encoder_dim (D): Dimension of encoder annotations
        decoder_dim (n): Dimension of GRU hidden state
        kernel_size: Kernel size for coverage convolution (Q in paper)
        """
        super(AttentionWithCoverage, self).__init__()
        self.attention_dim = attention_dim # n'
        self.encoder_dim = encoder_dim # D
        self.decoder_dim = decoder_dim # n
        # Attention MLP parameters
        # W_a: transforms previous hidden state h_{t-1}
        self.W_a = nn.Linear(decoder_dim, attention_dim, bias=False) # n' x n
        # U_a: transforms annotation vectors a_i
        self.U_a = nn.Linear(encoder_dim, attention_dim, bias=False) # n' x D
        # U_f: transforms coverage vector f_i
        self.U_f = nn.Linear(kernel_size, attention_dim, bias=False) # n' x kernel_size
        # v_a: attention scoring vector
        self.v_a = nn.Linear(attention_dim, 1, bias=False) # n' -> 1
        # Coverage convolution layer Q
        self.coverage_conv = nn.Conv1d(
            in_channels=1,
            out_channels=kernel_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

    def forward(self, annotations, h_prev, beta_prev):
        """
        Compute attention weights and context vector.
        Args:
        annotations: Encoder output (batch, L, D) where L = H x W
        h_prev: Previous GRU hidden state (batch, n)
        beta_prev: Sum of past attention probabilities (batch, L)
        Returns:
        context: Context vector c_t (batch, D)
        alpha: Attention weights (batch, L)
        beta: Updated cumulative attention (batch, L)
        """
        batch_size = annotations.size(0)
        L = annotations.size(1) # Number of annotation vectors
        # Compute coverage vectors F = Q * beta_t
        # beta_prev: (batch, L) -> (batch, 1, L) for conv1d
        beta_expanded = beta_prev.unsqueeze(1) # (batch, 1, L)
        F = self.coverage_conv(beta_expanded) # (batch, kernel_size, L)
        F = F.permute(0, 2, 1) # (batch, L, kernel_size)
        # Compute attention energies: e_ti = v_a^T * tanh(W_a * h_{t-1} + U_a * a_i + U_f * f_i)
        # Transform h_prev: (batch, n) -> (batch, 1, n') -> (batch, L, n')
        h_transformed = self.W_a(h_prev).unsqueeze(1).expand(batch_size, L, self.attention_dim)
        # Transform annotations: (batch, L, D) -> (batch, L, n')
        a_transformed = self.U_a(annotations)
        # Transform coverage vectors: (batch, L, kernel_size) -> (batch, L, n')
        f_transformed = self.U_f(F)
        # Combine and apply tanh
        energy_input = h_transformed + a_transformed + f_transformed # (batch, L, n')
        energy = self.v_a(torch.tanh(energy_input)).squeeze(2) # (batch, L)
        # Compute attention weights using softmax
        alpha = torch.softmax(energy, dim=1) # (batch, L)
        # Update cumulative attention: beta_t = sum_{l=1}^{t-1} alpha_l
        # For next timestep: beta_next = beta_prev + alpha
        beta = beta_prev + alpha # (batch, L)
        # Compute context vector: c_t = sum_{i=1}^L (alpha_ti * a_i)
        # (batch, L, 1) * (batch, L, D) -> (batch, L, D) -> (batch, D)
        context = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1) # (batch, D)
        return context, alpha, beta

class GRUDecoder(nn.Module):
    """
    GRU-based decoder with attention mechanism and coverage vector.
    """
    def __init__(self, vocab_size, embedding_dim, decoder_dim, encoder_dim, attention_dim, kernel_size=11):
        """
        Args:
        vocab_size (K): Number of words in vocabulary
        embedding_dim (m): Dimension of word embeddings (256)
        decoder_dim (n): Dimension of GRU hidden state (256)
        encoder_dim (D): Dimension of encoder annotations
        attention_dim (n'): Dimension of attention network
        kernel_size: Kernel size for coverage convolution
        """
        super(GRUDecoder, self).__init__()
        self.vocab_size = vocab_size # K
        self.embedding_dim = embedding_dim # m = 256
        self.decoder_dim = decoder_dim # n = 256
        self.encoder_dim = encoder_dim # D
        # Embedding matrix E: K x m
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU parameters for update gate z_t
        self.W_yz = nn.Linear(embedding_dim, decoder_dim, bias=False) # W_yz: n x m
        self.U_hz = nn.Linear(decoder_dim, decoder_dim, bias=False) # U_hz: n x n
        self.C_cz = nn.Linear(encoder_dim, decoder_dim, bias=False) # C_cz: n x D
        # GRU parameters for reset gate r_t
        self.W_yr = nn.Linear(embedding_dim, decoder_dim, bias=False) # W_yr: n x m
        self.U_hr = nn.Linear(decoder_dim, decoder_dim, bias=False) # U_hr: n x n
        self.C_cr = nn.Linear(encoder_dim, decoder_dim, bias=False) # C_cr: n x D
        # GRU parameters for candidate activation h_tilde
        self.W_yh = nn.Linear(embedding_dim, decoder_dim, bias=False) # W_yh: n x m
        self.U_rh = nn.Linear(decoder_dim, decoder_dim, bias=False) # U_rh: n x n
        self.C_ch = nn.Linear(encoder_dim, decoder_dim, bias=False) # C_ch: n x D
        # Attention mechanism with coverage
        self.attention = AttentionWithCoverage(attention_dim, encoder_dim, decoder_dim, kernel_size)
        # Output MLP parameters
        # p(y_t | x, y_{t-1}) = g(W_o * (E*y_{t-1} + W_h*h_t + W_c*c_t))
        self.W_h = nn.Linear(decoder_dim, embedding_dim, bias=False) # W_h: m x n
        self.W_c = nn.Linear(encoder_dim, embedding_dim, bias=False) # W_c: m x D
        self.W_o = nn.Linear(embedding_dim, vocab_size, bias=False) # W_o: K x m

    def forward_step(self, y_prev, h_prev, c_prev, annotations, beta_prev):
        """
        Single decoding step.
        Args:
        y_prev: Previous target word (batch,) - word indices
        h_prev: Previous GRU hidden state (batch, n)
        c_prev: Previous context vector (batch, D)
        annotations: Encoder output (batch, L, D)
        beta_prev: Cumulative attention from previous steps (batch, L)
        Returns:
        prob: Output word probability distribution (batch, K)
        h_t: Current GRU hidden state (batch, n)
        c_t: Current context vector (batch, D)
        alpha: Current attention weights (batch, L)
        beta_t: Updated cumulative attention (batch, L)
        """
        batch_size = annotations.size(0)
        # Compute context vector c_t using attention with coverage
        c_t, alpha, beta_t = self.attention(annotations, h_prev, beta_prev)
        # Embed previous word: y_{t-1} -> E*y_{t-1}
        y_embedded = self.embedding(y_prev) # (batch, m)
        # GRU computation
        # Update gate: z_t = σ(W_yz * E*y_{t-1} + U_hz * h_{t-1} + C_cz * c_t)
        z_t = torch.sigmoid(
            self.W_yz(y_embedded) + self.U_hz(h_prev) + self.C_cz(c_t)
        ) # (batch, n)
        # Reset gate: r_t = σ(W_yr * E*y_{t-1} + U_hr * h_{t-1} + C_cr * c_t)
        r_t = torch.sigmoid(
            self.W_yr(y_embedded) + self.U_hr(h_prev) + self.C_cr(c_t)
        ) # (batch, n)
        # Candidate activation: h_tilde = tanh(W_yh * E*y_{t-1} + U_rh * (r_t ⊙ h_{t-1}) + C_ch * c_t)
        h_tilde = torch.tanh(
            self.W_yh(y_embedded) + self.U_rh(r_t * h_prev) + self.C_ch(c_t)
        ) # (batch, n)
        # Hidden state update: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
        h_t = (1 - z_t) * h_prev + z_t * h_tilde # (batch, n)
        # Compute output probability: p(y_t | x, y_{t-1}) = g(W_o * (E*y_{t-1} + W_h*h_t + W_c*c_t))
        output_input = y_embedded + self.W_h(h_t) + self.W_c(c_t) # (batch, m)
        logits = self.W_o(output_input) # (batch, K)
        prob = F.log_softmax(logits, dim=1) # (batch, K)
        return prob, h_t, c_t, alpha, beta_t

    def forward(self, annotations, targets, teacher_forcing_ratio=1.0):
        """
        Forward pass for training.
        Args:
        annotations: Encoder output (batch, L, D) where L = H x W
        targets: Target sequences (batch, max_len) - word indices
        teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
        outputs: Predicted probabilities (batch, max_len, K)
        attentions: Attention weights for visualization (batch, max_len, L)
        """
        batch_size = annotations.size(0)
        L = annotations.size(1)
        max_len = targets.size(1)
        # Initialize hidden state, context vector, and beta (coverage tracking)
        h_t = torch.zeros(batch_size, self.decoder_dim, device=annotations.device)
        c_t = torch.zeros(batch_size, self.encoder_dim, device=annotations.device)
        beta_t = torch.zeros(batch_size, L, device=annotations.device)
        # Storage for outputs and attention weights
        outputs = []
        attentions = []
        # Start with <START> token (assume index 1, adjust if different)
        y_t = targets[:, 0] # First token is <START>
        for t in range(1, max_len):
            # Forward step
            prob, h_t, c_t, alpha, beta_t = self.forward_step(
                y_t, h_t, c_t, annotations, beta_t
            )
            outputs.append(prob.unsqueeze(1))
            attentions.append(alpha.unsqueeze(1))
            # Teacher forcing: use ground truth with probability teacher_forcing_ratio
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                y_t = targets[:, t]
            else:
                y_t = prob.argmax(dim=1)
        outputs = torch.cat(outputs, dim=1) # (batch, max_len-1, K)
        attentions = torch.cat(attentions, dim=1) # (batch, max_len-1, L)
        return outputs, attentions

    def decode_beam_search(self, annotations, start_token, end_token, max_len=150, beam_width=10):
        """
        Beam search decoding for inference.
        Args:
            annotations: Encoder output (1, L, D) - single image
            start_token: Start token index
            end_token: End token index
            max_len: Maximum sequence length
            beam_width: Beam width for search
        Returns:
            best_sequence: Best predicted sequence
            attention_weights: Attention weights for visualization
        """
        device = annotations.device
        L = annotations.size(1)
    
        # Initialize beam
        # Each beam element: (sequence, log_prob, h_t, c_t, beta_t, attentions)
        beams = [(
            [start_token],
            0.0,
            torch.zeros(1, self.decoder_dim, device=device),
            torch.zeros(1, self.encoder_dim, device=device),
            torch.zeros(1, L, device=device),
            []
        )]
    
        completed = []

        for _ in range(max_len):
            candidates = []
            for seq, score, h_t, c_t, beta_t, attn_list in beams:
                if seq[-1] == end_token:
                    # Convert to 3-element tuple for completed
                    completed.append((seq, score, attn_list))
                    continue
    
                # Get last token
                y_t = torch.tensor([seq[-1]], dtype=torch.long, device=device)
    
                # Forward step
                prob, h_new, c_new, alpha, beta_new = self.forward_step(
                    y_t, h_t, c_t, annotations, beta_t
                )
    
                # Get top-k predictions
                log_probs, tokens = prob[0].topk(beam_width)
    
                for log_prob, token in zip(log_probs, tokens):
                    new_seq = seq + [token.item()]
                    new_score = score + log_prob.item()
                    new_attn = attn_list + [alpha[0].cpu().numpy()]
    
                    candidates.append((
                        new_seq, new_score, h_new, c_new, beta_new, new_attn
                    ))
    
            # Select top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            # Stop if all beams are completed
            if len(beams) == 0:
                break
    
        # Add remaining beams to completed (convert to 3-element tuples)
        for seq, score, h_t, c_t, beta_t, attn_list in beams:
            completed.append((seq, score, attn_list))
    
        # Select best sequence
        completed.sort(key=lambda x: x[1] / len(x[0]), reverse=True)  # Normalize by length
        best_sequence, _, attention_weights = completed[0]
        return best_sequence, attention_weights