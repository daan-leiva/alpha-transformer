"""
Positional encoding utilities.

This module defines:
1. SinusoidalPositionalEncoding for fixed, non trainable position signals.
2. LearnablePositionalEncoding for trainable position embeddings.

Both modules expect input of shape (batch_size, seq_len, d_model) and return
the same shape with positional information added.
"""

import torch.nn as nn
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements fixed sinusoidal positional encodings, as described in the original Transformer paper.

    These encodings are not learned and allow the model to extrapolate to longer
    sequences at inference time, as long as max_len was large enough during construction.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Parameters
        ----------
        d_model : int
            Dimension of embeddings.
        max_len : int
            Maximum sequence length supported.
        """
        super().__init__()

        # Create a (max_len, d_model) tensor filled with zeros
        positional_encoding = torch.zeros(max_len, d_model)

        # Generate position indices (0 to max_len-1), shape: (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)

        # Compute inverse frequency scaling terms (for even dimensions)
        # Avoid large exponentiation instability by using log-exp trick
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Apply sine to even indices: 0, 2, 4, ...
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices: 1, 3, 5, ...
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)

        # Register as buffer so it is moved with the module to the correct device,
        # saved in state dict, and not treated as a trainable parameter.
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        """
        Add sinusoidal positional encodings to the input embeddings.

        Parameters
        ----------
        x : Tensor
            Input embeddings of shape (batch_size, seq_len, d_model).

        Returns
        -------
        Tensor
            Tensor of the same shape as x with positional encodings added.
        """
        seq_len = x.size(1)
        return x + self.positional_encoding[:, :seq_len, :]

class LearnablePositionalEncoding(nn.Module):
    """
    Implements learnable positional encodings using a trainable embedding matrix.

    Each position from 0 to max_len gets a learnable vector that is added to the
    token embedding at that position.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Parameters
        ----------
        d_model : int
            Dimension of embeddings.
        max_len : int
            Maximum sequence length supported.
        """
        super().__init__()

        # Learnable embeddings for each position index [0, max_len)
        self.position_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)

    def forward(self, x):
        """
        Add learnable positional embeddings to the input.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        Tensor
            Tensor of shape (batch_size, seq_len, d_model) with position embeddings added.
        """
        batch_size, seq_len, _ = x.size()

        # Create position indices for each time step in the sequence
        # positions shape: (batch_size, seq_len)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Lookup embeddings and add to input
        return x + self.position_embedding(positions)
