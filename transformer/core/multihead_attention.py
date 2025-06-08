import torch.nn as nn
from transformer.core.singlehead_attention import SingleAttentionHead
import torch

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
        self.q_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_projection = nn.Linear(in_features=d_model, out_features=d_model)
        self.single_attention = SingleAttentionHead(dropout_rate=dropout_rate)
        self.recombination = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, Q, K, V, mask=None, past_key_value=None):
        # get other x dimensions
        batch_size, src_seq_len, _ = K.shape # get them separate for the case of cross attention
        batch_size, target_seq_len, _ = Q.shape

        # project X into a bigger space for a later split
        Q = self.q_projection(Q)
        K = self.k_projection(K)
        V = self.v_projection(V)
        # reshape for single attention
        # permute dimensions so we perform single attention per head instead of per sequence
        Q = Q.reshape(batch_size, target_seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.reshape(batch_size, src_seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.reshape(batch_size, src_seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Concatenate past key/value if they exist
        # this will be used during inference
        # only the new seq value will be passed intead of the whole sequence
        # this will keep all dimensions cohesive
        if past_key_value is not None:
            past_K, past_V = past_key_value  # Each is (batch_size, n_heads, past_seq_len, d_k)
            # Concatenate along sequence length dimension (dim=2)
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        # save updated (key, value) for use in following iterations
        new_past_key_value = (K, V)

        # perform single attention
        weighted_value_matrix, weighted_attention_matrix = self.single_attention(Q=Q, K=K, V=V, mask=mask)

        # switch the seq_len dimension back to the second position
        weighted_value_matrix = weighted_value_matrix.permute(0, 2, 1, 3)
        # combine n_heads and d_k into d_model
        weighted_value_matrix = weighted_value_matrix.reshape(batch_size, target_seq_len, self.d_model)
        # recombine the data through learned weights
        output = self.recombination(weighted_value_matrix)

        return output, weighted_attention_matrix, new_past_key_value