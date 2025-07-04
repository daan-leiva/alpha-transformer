import torch.nn as nn
from transformer.core.multihead_attention import MultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    """
    Implements a single decoder block for the Transformer model.
    Includes:
        - masked self-attention
        - cross-attention over encoder output
        - position-wise feedforward network
        - residual connections and layer normalization
    """

    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d):
        """
        Args:
            d_model (int): Dimensionality of model embeddings
            n_heads (int): Number of attention heads
            dropout_rate (float): Dropout rate
            hidden_ff_d (int): Hidden dimension of the feedforward sublayer
        """
        super().__init__()

        # --- Self-Attention Block ---
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        # --- Cross-Attention Block ---
        # Q comes from decoder, K and V from encoder output
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

        # --- Feedforward Block ---
        # Applies two linear layers with ReLU activation and dropout in between
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, hidden_ff_d),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_ff_d, d_model)
        )
        self.norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None,
                past_key_value=None, return_attention=False):
        """
        Forward pass through the decoder block.

        Args:
            x (Tensor): Input tensor (batch_size, tgt_len, d_model)
            encoder_output (Tensor): Output from encoder (batch_size, src_len, d_model)
            tgt_mask (Tensor): Mask for decoder self-attention (batch_size, 1, tgt_len, tgt_len)
            src_mask (Tensor): Mask for encoder-decoder attention (batch_size, 1, 1, src_len)
            past_key_value (Tuple): Optional tuple of cached (K, V) from previous decoding step
            return_attention (bool): Whether to return attention weights from cross-attention

        Returns:
            x (Tensor): Output after decoder block (batch_size, tgt_len, d_model)
            new_past_key_value (Tuple): Cached (K, V) from self-attention for use in autoregressive decoding
            weighted_attention_matrix (Tensor, optional): Cross-attention weights if requested
        """

        # --- Masked Self-Attention + Residual ---
        self_attn_output, _, new_past_key_value = self.self_attention(
            Q=x, K=x, V=x, mask=tgt_mask, past_key_value=past_key_value
        )
        x = self.norm1(x + self.dropout1(self_attn_output))

        # --- Cross-Attention (Decoder-Encoder) + Residual ---
        cross_attn_output, weighted_attention_matrix, _ = self.cross_attention(
            Q=x, K=encoder_output, V=encoder_output, mask=src_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # --- Feedforward Network + Residual ---
        ff_output = self.feedforward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        if return_attention:
            return x, new_past_key_value, weighted_attention_matrix
        else:
            return x, new_past_key_value
