import torch.nn as nn
from core.positional_encoding import SinusoidalPositionalEncoding, LearnablePositionalEncoding
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout_rate, encoding_type='sinusoidal'):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # chose desired positional encoding layer
        if encoding_type == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        elif encoding_type == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(d_model, max_len)
        else:
            raise ValueError("Invalid encoding type selected")

        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # scale for consistent sizing for the embeddings vs positional encoding
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        x = self.token_embedding(x) * self.scale
        x = self.positional_encoding(x)
        x = self.dropout(x)

        return x