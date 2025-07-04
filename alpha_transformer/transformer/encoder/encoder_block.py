import torch.nn as nn
from transformer.core.multihead_attention import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    """
    A single encoder block consisting of:
      - Multi-head self-attention
      - Layer normalization and residual connections
      - Feedforward neural network (FFN)
    
    Args:
        d_model (int): Embedding dimension
        n_heads (int): Number of attention heads
        hidden_ff_d (int): Hidden dimension in feedforward network
        dropout_rate (float): Dropout rate (default: 0.1)
    """
    def __init__(self, d_model, n_heads, hidden_ff_d, dropout_rate=0.1):
        super().__init__()

        # Multi-head self-attention layer
        self.multi_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate
        )

        # First layer normalization (applied after residual connection)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)

        # Dropout applied after attention
        self.dropout1 = nn.Dropout(dropout_rate)

        # Position-wise feedforward network with ReLU activation
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_ff_d),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ff_d, d_model)
        )

        # Second layer normalization and dropout
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        """
        Forward pass through the encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (Tensor): Attention mask of shape (batch_size, 1, 1, seq_len)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # --- Multi-head self-attention ---
        # Q, K, V are all x (self-attention)
        self_attention_output, _, _ = self.multi_attention(Q=x, K=x, V=x, mask=mask)

        # Add & Norm (residual connection + normalization)
        x = self.norm1(x + self.dropout1(self_attention_output))

        # --- Feedforward Network ---
        ff_input = x
        x = self.ff(x)

        # Add & Norm again
        x = self.norm2(ff_input + self.dropout2(x))

        return x
