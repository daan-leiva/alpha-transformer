import torch.nn as nn
from alpha_transformer.transformer.core.multihead_attention import MultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    """
    Single decoder block for the Transformer model.

    The block contains:
      masked self attention over the decoder input
      cross attention over encoder outputs
      positionwise feedforward network
      residual connections and layer normalization around each sublayer

    It also supports cached keys and values in the self attention sublayer
    to enable efficient autoregressive decoding.
    """

    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d):
        """
        Parameters
        ----------
        d_model : int
            Dimensionality of model embeddings.
        n_heads : int
            Number of attention heads.
        dropout_rate : float
            Dropout probability.
        hidden_ff_d : int
            Hidden dimension of the feedforward sublayer.
        """
        super().__init__()

        # Self attention block on the decoder side
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Cross attention block where queries come from the decoder and
        # keys and values come from the encoder output
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Positionwise feedforward network
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

        Parameters
        ----------
        x : Tensor
            Decoder input of shape (batch_size, tgt_len, d_model).
        encoder_output : Tensor
            Encoder output of shape (batch_size, src_len, d_model).
        tgt_mask : Tensor, optional
            Mask for decoder self attention of shape
            (batch_size, 1, tgt_len, tgt_len). Typically combines a causal
            mask with padding masks.
        src_mask : Tensor, optional
            Mask for cross attention of shape (batch_size, 1, 1, src_len).
            Positions with value 0 are excluded from attention.
        past_key_value : tuple(Tensor, Tensor), optional
            Cached self attention key and value from previous decoding steps.
            Each tensor has shape (batch_size, n_heads, past_len, d_k).
        return_attention : bool
            If True, also returns cross attention weights.

        Returns
        -------
        x : Tensor
            Output after the decoder block of shape (batch_size, tgt_len, d_model).
        new_past_key_value : tuple(Tensor, Tensor)
            Updated cached self attention key and value to use in the next step.
        weighted_attention_matrix : Tensor, optional
            Cross attention weights of shape
            (batch_size, n_heads, tgt_len, src_len) if return_attention is True.
        """
        # Masked self attention with optional cached K and V for autoregressive decoding
        self_attn_output, _, new_past_key_value = self.self_attention(
            Q=x, K=x, V=x, mask=tgt_mask, past_key_value=past_key_value
        )
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross attention over encoder outputs, used for source to target alignment
        cross_attn_output, weighted_attention_matrix, _ = self.cross_attention(
            Q=x, K=encoder_output, V=encoder_output, mask=src_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Positionwise feedforward sublayer
        ff_output = self.feedforward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        if return_attention:
            return x, new_past_key_value, weighted_attention_matrix
        else:
            return x, new_past_key_value
