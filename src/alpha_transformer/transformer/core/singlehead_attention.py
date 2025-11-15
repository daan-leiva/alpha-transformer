import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleAttentionHead(nn.Module):
    """
    Single scaled dot product attention head.

    This module assumes that the head dimension is already present in the input
    tensors and works on tensors with shape:

        (batch_size, n_heads, seq_len, d_k)

    It does not perform any projection of Q, K, or V. Those projections are
    handled by MultiHeadAttention.
    """

    def __init__(self, dropout_rate=0.1):
        """
        Parameters
        ----------
        dropout_rate : float
            Dropout probability applied to the attention weights.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for a single attention head.

        Parameters
        ----------
        Q : Tensor
            Query tensor of shape (batch_size, n_heads, seq_len_q, d_k).
        K : Tensor
            Key tensor of shape   (batch_size, n_heads, seq_len_k, d_k).
        V : Tensor
            Value tensor of shape (batch_size, n_heads, seq_len_k, d_k).
        mask : Tensor, optional
            Attention mask of shape (batch_size, 1, seq_len_q, seq_len_k) or
            broadcastable to that shape. Entries with value 0 are masked out
            before softmax. Entries with value 1 are kept.

        Returns
        -------
        weighted_value_matrix : Tensor
            Attention output of shape (batch_size, n_heads, seq_len_q, d_k).
        weighted_attention_matrix : Tensor
            Normalized attention weights of shape
            (batch_size, n_heads, seq_len_q, seq_len_k).
        """
        # Head depth used for scaling
        d_k = Q.size(-1)

        # Scale factor reduces the magnitude of dot products
        # and keeps the softmax in a numerically stable range.
        scale = 1.0 / torch.sqrt(torch.tensor(float(d_k), device=Q.device, dtype=Q.dtype))

        # Raw attention scores by dot product between queries and keys
        # Resulting shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        attention_matrix = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply mask for padding or causal constraints
        # Mask entries with value 0 become large negative numbers so that
        # softmax drives them close to zero.
        if mask is not None:
            attention_matrix = attention_matrix.masked_fill(mask == 0, float('-inf'))

        # Normalize scores along the key dimension
        weighted_attention_matrix = F.softmax(attention_matrix, dim=-1)

        # Regularize by dropping some attention mass
        weighted_attention_matrix = self.dropout(weighted_attention_matrix)

        # Weighted sum of values using attention weights
        # Shape: (batch_size, n_heads, seq_len_q, d_k)
        weighted_value_matrix = torch.matmul(weighted_attention_matrix, V)

        return weighted_value_matrix, weighted_attention_matrix
