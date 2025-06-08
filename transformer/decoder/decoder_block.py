import torch.nn as nn
from transformer.core.multihead_attention import MultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d):
        super().__init__()
        # self attention
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        # cross attention (mixes the query from the decoder with the key and value from the encoder)
        # allows the decoder to focus on important parts of the encoder
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

        # postition wise feedforward network
        self.feedforward = nn.Sequential(nn.Linear(d_model, hidden_ff_d),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_rate),
                                        nn.Linear(hidden_ff_d, d_model))
        # normalize and dropout
        self.norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout3 = nn.Dropout(dropout_rate)

    # input shape : (batch_size, tgt_len, d_model)
    # encoder_output shape : (batch_size, src_len, d_model)
    # tgt_mask shape : (batch_size, 1, tgt_len, tgt_len) [masks the self attention matrix]
    # source_mask shape : (batch_size, 1, 1, src_len) [mask the encoder output]
    # tgt_len = the sequence we want the encoder to be exposed to at this point in time
    # (a subset of the seq_len)
    # src_len = the the length of the encoder sequence length
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None,
                past_key_value=None):
        # self attention block + residual connection
        self_attention_output, _, new_past_key_value = self.self_attention(Q=x, K=x, V=x,
                                                       mask=tgt_mask,
                                                       past_key_value=past_key_value)
        x = self.norm1(x + self.dropout1(self_attention_output))

        # cross attention block + residual connection
        cross_attention_matrix, _, _ = self.cross_attention(Q=x, K=encoder_output, V=encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_attention_matrix))

        # feedforward + residual
        ff_out = self.feedforward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x, new_past_key_value