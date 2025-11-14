import torch.nn as nn
from transformer.core.multihead_attention import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    """
    Single encoder block composed of:

      multi head self attention
      residual connection and layer normalization
      positionwise feedforward network
      residual connection and layer normalization

    This is the standard encoder block structure used in the original
    Transformer architecture.
    """
    def __init__(self, d_model, n_heads, hidden_ff_d, dropout_rate=0.1):
        """
        Parameters
        ----------
        d_model : int
            Embedding dimension and model width.
        n_heads : int
            Number of attention heads.
        hidden_ff_d : int
            Hidden dimension in the feedforward network.
        dropout_rate : float
            Dropout probability used in attention and feedforward layers.
        """
        super().__init__()

        # Multi-head self-attention layer
        self.multi_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate
        )

        # Layer normalization applied after residual self attention
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Positionwise feedforward network with nonlinearity
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_ff_d),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ff_d, d_model)
        )

        # Layer normalization applied after residual feedforward
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        """
        Forward pass through the encoder block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        mask : Tensor
            Attention mask of shape (batch_size, 1, 1, seq_len).
            Mask entries with value 0 indicate positions that should not
            receive attention (for example padding tokens).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self attention uses x as queries, keys, and values
        self_attention_output, _, _ = self.multi_attention(Q=x, K=x, V=x, mask=mask)

        # Residual connection followed by normalization
        # x + attention_output preserves the input path while integrating
        # the attention output.
        x = self.norm1(x + self.dropout1(self_attention_output))

        # Positionwise feedforward network applied to each position
        ff_input = x
        x = self.ff(x)

        # Second residual connection and normalization
        x = self.norm2(ff_input + self.dropout2(x))

        return x
