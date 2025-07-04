import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleAttentionHead(nn.Module):
    """
    Implements a single scaled dot-product attention head.
    """

    def __init__(self, dropout_rate=0.1):
        """
        Args:
            dropout_rate (float): Dropout applied to the attention weights.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for a single attention head.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, n_heads, seq_len_q, d_k)
            K (Tensor): Key tensor of shape   (batch_size, n_heads, seq_len_k, d_k)
            V (Tensor): Value tensor of shape (batch_size, n_heads, seq_len_k, d_k)
            mask (Tensor, optional): Attention mask of shape (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            weighted_value_matrix (Tensor): Output after applying attention, same shape as Q
            weighted_attention_matrix (Tensor): Attention weights, shape (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        d_k = Q.size(-1)  # Dimensionality of each query/key vector

        # Scale factor to prevent extremely large dot products
        scale = 1.0 / torch.sqrt(torch.tensor(float(d_k), device=Q.device, dtype=Q.dtype))

        # Compute raw attention scores by scaled dot-product
        # Output shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        attention_matrix = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply the attention mask (e.g. for padding or causal masking)
        if mask is not None:
            attention_matrix = attention_matrix.masked_fill(mask == 0, float('-inf'))

        # Normalize the attention scores across the key dimension using softmax
        weighted_attention_matrix = F.softmax(attention_matrix, dim=-1)

        # Apply dropout to the attention weights (helps regularize)
        weighted_attention_matrix = self.dropout(weighted_attention_matrix)

        # Use attention weights to blend the values
        # Output shape: (batch_size, n_heads, seq_len_q, d_k)
        weighted_value_matrix = torch.matmul(weighted_attention_matrix, V)

        return weighted_value_matrix, weighted_attention_matrix
