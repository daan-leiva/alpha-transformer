import torch.nn as nn
from alpha_transformer.transformer.core.positional_encoding import SinusoidalPositionalEncoding, LearnablePositionalEncoding
import math

class InputEmbedding(nn.Module):
    """
    Combine token embeddings and positional encodings to produce model inputs.

    This is used on both encoder and decoder sides. It maps integer token ids
    to dense embeddings, scales them, adds positional information, and applies
    dropout.
    """

    def __init__(self, vocab_size, d_model, max_len, dropout_rate, encoding_type='sinusoidal'):
        """
        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        d_model : int
            Embedding dimension and model width.
        max_len : int
            Maximum supported sequence length.
        dropout_rate : float
            Dropout probability applied after adding positional encodings.
        encoding_type : str
            Type of positional encoding, either "sinusoidal" or "learnable".
        """
        super().__init__()

        # Token embedding maps each token id to a dense vector
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        # Positional encoding adds sequence position information
        if encoding_type == 'sinusoidal':
            self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        elif encoding_type == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(d_model=d_model, max_len=max_len)
        else:
            raise ValueError("Invalid encoding type selected. Choose 'sinusoidal' or 'learnable'.")

        # Dropout for regularization after combining token and position information
        self.dropout = nn.Dropout(p=dropout_rate)

        # Scale embeddings as suggested in the Transformer paper
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Input token ids of shape (batch_size, seq_len).

        Returns
        -------
        Tensor
            Embedded input of shape (batch_size, seq_len, d_model).
        """
        # Embed tokens and scale by sqrt(d_model)
        x = self.token_embedding(x) * self.scale

        # Add positional encodings
        x = self.positional_encoding(x)

        # Apply dropout and retur
        x = self.dropout(x)
        
        return x
