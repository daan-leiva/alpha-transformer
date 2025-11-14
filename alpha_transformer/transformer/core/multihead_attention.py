"""
Multi head attention block built on top of a single attention head.

This module:
1. Projects input queries, keys, and values into head specific subspaces.
2. Applies scaled dot product attention in parallel across heads.
3. Concatenates the head outputs and projects back to the model dimension.
4. Supports cached keys and values for incremental decoding.
"""

import torch.nn as nn
from transformer.core.singlehead_attention import SingleAttentionHead
import torch


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention by projecting input queries, keys, and values into
    multiple heads, performing scaled dot-product attention independently, and then
    recombining the results into the original dimensionality.
    """

    def __init__(self, d_model, n_heads, dropout_rate):
        """
        Parameters
        ----------
        d_model : int
            Dimensionality of input and output features.
        n_heads : int
            Number of attention heads.
        dropout_rate : float
            Dropout rate used inside each attention head.
        """
        super().__init__()

        # Ensure d_model is divisible by number of heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Save configuration
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimensionality per head

        # Linear projections for Q, K, V
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)

        # Shared attention logic across all heads
        # This module operates on tensors that already have a head dimension
        self.single_attention = SingleAttentionHead(dropout_rate=dropout_rate)

        # Output projection to collapse all heads back into d_model
        self.recombination = nn.Linear(d_model, d_model)


    def forward(self, Q, K, V, mask=None, past_key_value=None):
        """
        Multi head attention forward pass.

        Parameters
        ----------
        Q : Tensor
            Query tensor of shape (batch_size, tgt_seq_len, d_model).
        K : Tensor
            Key tensor of shape (batch_size, src_seq_len, d_model).
        V : Tensor
            Value tensor of shape (batch_size, src_seq_len, d_model).
        mask : Tensor, optional
            Broadcastable attention mask, for example (batch_size, 1, 1, src_seq_len)
            or (batch_size, 1, tgt_seq_len, src_seq_len).
        past_key_value : tuple(Tensor, Tensor), optional
            Cached key and value tensors used during incremental decoding.
            Each has shape (batch_size, n_heads, past_seq_len, d_k).

        Returns
        -------
        output : Tensor
            Attention output of shape (batch_size, tgt_seq_len, d_model).
        attention_weights : Tensor
            Attention scores of shape (batch_size, n_heads, tgt_seq_len, src_total_len)
            where src_total_len includes any cached positions when used.
        new_past_key_value : tuple(Tensor, Tensor)
            Updated (K, V) pair with the current step appended, to be reused
            in the next decoding step.
        """
        batch_size, src_seq_len, _ = K.shape
        _, tgt_seq_len, _ = Q.shape

        # Project Q, K, V from input model space into head mixed space
        Q = self.q_projection(Q)  # (batch_size, tgt_seq_len, d_model)
        K = self.k_projection(K)  # (batch_size, src_seq_len, d_model)
        V = self.v_projection(V)  # (batch_size, src_seq_len, d_model)

        # Reshape and split into heads
        # After reshape: (batch_size, seq_len, n_heads, d_k)
        # After permute: (batch_size, n_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, tgt_seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, src_seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, src_seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Concatenate past keys and values if available (for autoregressive decoding)
        if past_key_value is not None:
            past_K, past_V = past_key_value  # Each: (batch_size, n_heads, past_seq_len, d_k)
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        # Save new past key/value for future use in autoregressive decoding
        new_past_key_value = (K, V)

        # Apply attention across all heads
        # weighted_value_matrix: (batch_size, n_heads, tgt_seq_len, d_k)
        # attention_weights:     (batch_size, n_heads, tgt_seq_len, src_total_len)
        weighted_value_matrix, attention_weights = self.single_attention(Q=Q, K=K, V=V, mask=mask)

        # Recombine attention outputs:
        # Permute to (batch_size, tgt_seq_len, n_heads, d_k)
        weighted_value_matrix = weighted_value_matrix.permute(0, 2, 1, 3)
        # Collapse heads into d_model: (batch_size, tgt_seq_len, d_model)
        weighted_value_matrix = weighted_value_matrix.reshape(batch_size, tgt_seq_len, self.d_model)

        #  Final linear layer back to model space (batch_size, tgt_seq_len, d_model)
        output = self.recombination(weighted_value_matrix)

        return output, attention_weights, new_past_key_value
