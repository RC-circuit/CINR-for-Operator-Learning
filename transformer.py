import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Standard Sinusoidal Positional Encoding.
        Fixed weights (not learnable).
        """
        super().__init__()

        # Create a long enough 'pe' matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply Sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply Cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a parameter), so it saves with state_dict
        # but is not updated by the optimizer.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor [Batch, Seq_Len, d_model]
        """
        # Add positional encoding to input (slice to current sequence length)
        # x.size(1) is the sequence length T
        x = x + self.pe[:, :x.size(1), :]
        return x

class LatentTransformer(nn.Module):
    def __init__(self, latent_dim, d_model=256, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 1. Project Continuous Latents -> Transformer Dimension
        # Since we don't have word indices, we use a Linear layer instead of nn.Embedding
        self.input_proj = nn.Linear(latent_dim, d_model)
        
        # 2. Positional Encoding (Fixed Sinusoidal)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # 3. The "Brain": Stack of Transformer Layers
        # usage of 'batch_first=True' is critical for [Batch, Seq, Dim] format
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout, 
            batch_first=True,
            norm_first=True # Often stabilizes training
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Projection
        # Projects back from d_model -> latent_dim
        self.output_proj = nn.Linear(d_model, latent_dim)
        
        self.d_model = d_model

    def _generate_causal_mask(self, sz, device):
        """
        Generates a mask to prevent attending to future positions.
        Returns a float mask: 0.0 for allowed, -inf for forbidden.
        """
        # creates a matrix of 1s (upper triangle is 1)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        
        # Fill: 0.0 (allowed) where mask is 1, -inf (forbidden) where mask is 0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src):
        """
        Args:
            src: [Batch, Seq_Len, Latent_Dim]
        Returns:
            output: [Batch, Seq_Len, Latent_Dim]
        """
        # A. Input Projection [B, T, D] -> [B, T, d_model]
        # We scale by sqrt(d_model) as per the original paper (helps with variance)
        x = self.input_proj(src) * math.sqrt(self.d_model)
        
        # B. Add Position Info
        x = self.pos_encoder(x)
        
        # C. Generate Causal Mask
        # We need a mask of size [T, T]
        T = src.shape[1]
        mask = self._generate_causal_mask(T, src.device)
        
        # D. Transformer Pass
        # Pass the mask so the model acts as a DECODER (autoregressive)
        x = self.transformer_encoder(x, mask=mask)
        
        # E. Output Projection
        output = self.output_proj(x)
        
        return output