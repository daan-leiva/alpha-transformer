import torch.nn as nn
from core.singlehead_attention import SingleAttentionHead

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super().__init__()
        # verify that the model dimension can be evenly split amongst the heads
        assert(d_model%n_heads == 0)

        # save class variables
        self.d_k = d_model//n_heads # with d_k == d_q (and in this project also d_k == d_v)
        self.d_model = d_model
        self.n_heads = n_heads

        # spread the input into q, k and v variables
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.single_attention = SingleAttentionHead(dropout_rate=dropout_rate)
        self.recombination = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # get other x dimensions
        batch_size, seq_len, _ = Q.shape # in this implementation K and V are the same shape as Q

        # project X into a bigger space for a later split
        Q = self.q_projection(Q)
        K = self.q_projection(K)
        V = self.q_projection(V)
        # reshape for single attention
        # permute dimensions so we perform single attention per head instead of per sequence
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # perform single attention
        weighted_value_matrix, weighted_attention_matrix = self.single_attention(Q, K, V, mask)

        # switch the seq_len dimension back to the second position
        weighted_value_matrix = weighted_value_matrix.permute(0, 2, 1, 3)
        # combine n_heads and d_k into d_model
        weighted_value_matrix = weighted_value_matrix.reshape(batch_size, seq_len, self.d_model)
        # recombine the data through learned weights
        output = self.recombination(weighted_value_matrix)

        return output, weighted_attention_matrix