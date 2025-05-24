import torch.nn as nn
from core.input_embedding import InputEmbedding
from encoder.transformer_encoder import TransformerEncoder

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, max_len, dropout_rate, encoding_type,
                 hidden_ff_d, num_encoder_layers):
        super().__init__()

        self.input_embedding = InputEmbedding(vocab_size = vocab_size, d_model = d_model,
                                              max_len = max_len, dropout_rate = dropout_rate,
                                              encoding_type = encoding_type)
        self.encoder_stack = TransformerEncoder(d_model = d_model, n_heads = n_heads,
                                                d_encoder_ff = d_encoder_ff,
                                                num_encoder_layers = num_encoder_layers,
                                                dropout_rate = dropout_rate)
        
    # input shape : (batch_size, seq_len) - tokenized input
    # mask shape: (batch_size, 1, 1, seq_len) - to be applied to the last dim
    # of the multi-attention Q * K.T [batch_size, n_heads, seq_len, seq_len]
    # output shape: (batch_size, seq_len, d_model)
    def forward(self, x, mask=None):
        x = self.input_embedding(x)
        x = self.encoder_stack(x, mask)

        return x
