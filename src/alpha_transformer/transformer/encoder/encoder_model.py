"""
Encoder side of the Transformer model.

This module wraps:
1. InputEmbedding for token and position embeddings on the source side.
2. TransformerEncoder, a stack of self attention and feedforward blocks.

It exposes a single TransformerEncoderModel class that returns encoder hidden states.
"""

import torch.nn as nn
from alpha_transformer.transformer.core.input_embedding import InputEmbedding
from alpha_transformer.transformer.encoder.transformer_encoder import TransformerEncoder


class TransformerEncoderModel(nn.Module):
    """
    Full Transformer encoder model that combines:
      - Token and positional embedding
      - Stack of encoder blocks with self attention and feedforward layers

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of tokens).
    d_model : int
        Dimensionality of the model (embedding size).
    n_heads : int
        Number of attention heads.
    max_len : int
        Maximum input sequence length.
    dropout_rate : float
        Dropout probability.
    encoding_type : str
        Type of positional encoding, either "sinusoidal" or "learnable".
    hidden_ff_d : int
        Hidden dimension for the feedforward network.
    num_encoder_layers : int
        Number of encoder blocks to stack.
    """
    def __init__(self, vocab_size, d_model, n_heads, max_len, dropout_rate, encoding_type,
                 hidden_ff_d, num_encoder_layers):
        """
        Construct the encoder side wrapper.

        Parameters
        ----------
        vocab_size : int
            Size of the source vocabulary.
        d_model : int
            Embedding and hidden dimension used throughout the encoder.
        n_heads : int
            Number of attention heads per encoder block.
        max_len : int
            Maximum supported sequence length for positional encodings.
        dropout_rate : float
            Dropout probability applied in embeddings and blocks.
        encoding_type : str
            Positional encoding strategy, either "sinusoidal" for fixed encodings
            or "learnable" for trainable position embeddings.
        hidden_ff_d : int
            Hidden dimension of the feedforward sublayers inside each block.
        num_encoder_layers : int
            Number of stacked encoder blocks.
        """
        super().__init__()

        # Token + positional embedding
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout_rate=dropout_rate,
            encoding_type=encoding_type
        )

        # Encoder stack (multiple TransformerEncoder blocks)
        self.encoder_stack = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            hidden_ff_d=hidden_ff_d,
            num_encoder_layers=num_encoder_layers,
            dropout_rate=dropout_rate
        )

    def forward(self, x, mask=None):
        """
        Encode a batch of token ids.

        Parameters
        ----------
        x : Tensor
            Input token ids of shape (batch_size, seq_len).
        mask : Tensor, optional
            Mask tensor of shape (batch_size, 1, 1, seq_len), used to mask padding tokens
            or apply attention constraints.

        Returns
        -------
        Tensor
            Encoder output of shape (batch_size, seq_len, d_model).
        """
        # Apply embedding + positional encoding + dropout
        x = self.input_embedding(x)

        # Pass through encoder blocks
        x = self.encoder_stack(x, mask)

        return x
