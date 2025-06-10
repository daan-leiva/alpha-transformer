import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleAttentionHead(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch_size, n_heads, seq_len, d_k)
        mask: (batch_size, 1, seq_len, seq_len)
        """
        # for this project d_k == d_v (and d_q)
        d_k = Q.shape[-1]

        # calcualte our attention matrix
        attention_matrix = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k) # resulting dimensions = (batch_size, n_heads, seq_len, seq_len)
        # the mask will be used to partially hide inputs during training (look ahead)
        # and for variable input lengths
        if mask is not None:
            attention_matrix = attention_matrix.masked_fill(mask == 0, float('-inf'))

        # apply a softmax function to the results
        weighted_attention_matrix = F.softmax(attention_matrix, dim=-1)
        # apply dropout layer
        weighted_attention_matrix = self.dropout(weighted_attention_matrix)

        # multiply times the value element
        weighted_value_matrix = torch.matmul(weighted_attention_matrix, V)

        return weighted_value_matrix, weighted_attention_matrix