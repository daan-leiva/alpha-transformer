import torch.nn as nn
from core.multihead_attention import MultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d):
        super().__init__()
        # self attention
        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        # cross attention (mixes the query from the decoder with the key and value from the encoder)
        # allows the decoder to focus on important parts of the decoder
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

        # postition wise feedforward network
        self.feedfoward = nn.Sequential(nn.Linear(d_model, hidden_ff_d),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(hidden_ff_d, d_model))
        # normalize and dropout
        self.norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout3 = nn.Dropout(dropout_rate)

    # input shape : (batch_size, target_len, d_model)
    # encoder_ouput shape : (batch_size, src_len, d_model)
    # target_mask shape : (batch_size, 1, tgt_len, tgt_len) [masks the self attention matrix0]
    # source_mask shape : (batch_size, 1, 1, src_len) [mask the encoder output]
    # target_len = the sequence we want the encoder to be exposed to at this point in time
    # (a subset of the seq_len)
    # src_len = the the length of the encoder sequence length
    def forward(self, x, encoder_output, target_mask=None, src_mask=None):
        # self attention block + residual connection
        self_attention_ouput, _ = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout1(self_attention_ouput))

        # cross attention block + residual connection
        cross_attention_matrix, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attention_matrix))

        # feedforward + residual
        ff_out = self.feedfoward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x