import torch.nn as nn
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements fixed sinusoidal positional encodings, as described in the original Transformer paper.
    These encodings are not learned and allow the model to extrapolate to longer sequences at inference time.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimension of embeddings.
            max_len (int): Maximum sequence length supported.
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

        # Register as buffer (non-trainable, saves with model)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Positional encoding added to input.
        """
        seq_len = x.size(1)
        return x + self.positional_encoding[:, :seq_len, :]

class LearnablePositionalEncoding(nn.Module):
    """
    Implements learnable positional encodings using a trainable embedding matrix.
    Each position from 0 to max_len gets a learnable vector.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Dimension of embeddings.
            max_len (int): Maximum sequence length supported.
        """
        super().__init__()

        # Learnable embeddings for each position
        self.position_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Input with positional embeddings added.
        """
        batch_size, seq_len, _ = x.size()

        # Create position indices for each time step
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Lookup embeddings and add to input
        return x + self.position_embedding(positions)
