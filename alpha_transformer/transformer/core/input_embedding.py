import torch.nn as nn
from transformer.core.positional_encoding import SinusoidalPositionalEncoding, LearnablePositionalEncoding
import math

class InputEmbedding(nn.Module):
    """
    This module combines token embeddings and positional encodings to create
    the final input embeddings used by the Transformer encoder or decoder.
    """

    def __init__(self, vocab_size, d_model, max_len, dropout_rate, encoding_type='sinusoidal'):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimensionality of the model (embedding size).
            max_len (int): Maximum sequence length supported.
            dropout_rate (float): Dropout rate applied after embedding.
            encoding_type (str): Type of positional encoding to use: 'sinusoidal' or 'learnable'.
        """
        super().__init__()

        # === Token Embedding Layer ===
        # Maps each token ID to a dense vector of size d_model
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # === Positional Encoding Layer ===
        # Adds positional information to the input embeddings
        if encoding_type == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        elif encoding_type == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(d_model=d_model, max_len=max_len)
        else:
            raise ValueError("Invalid encoding type selected. Choose 'sinusoidal' or 'learnable'.")

        # === Dropout ===
        # Applied after adding positional encodings to help with regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        # === Scaling Factor ===
        # Scales embeddings by sqrt(d_model) as recommended in the Transformer paper
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input token IDs of shape (batch_size, seq_len)

        Returns:
            Tensor: Embedded input of shape (batch_size, seq_len, d_model)
        """
        # Embed tokens and scale
        x = self.token_embedding(x) * self.scale

        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply dropout and return
        x = self.dropout(x)
        return x
