"""
High level Transformer model that combines encoder and decoder stacks.

This module exposes a single Transformer class that:
1. Encodes a source sequence.
2. Decodes a target sequence conditioned on the encoder output.
3. Projects decoder states to vocabulary logits.
"""

import torch.nn as nn
from alpha_transformer.transformer.encoder.encoder_model import TransformerEncoderModel
from alpha_transformer.transformer.decoder.decoder_model import TransformerDecoderModel


class Transformer(nn.Module):
    """
    Full Transformer model combining encoder and decoder components.

    This follows the architecture described in the paper "Attention is All You Need".

    Parameters
    ----------
    vocab_size : int
        Vocabulary size shared across encoder and decoder.
    d_model : int
        Dimensionality of embeddings and hidden layers.
    n_heads : int
        Number of attention heads.
    max_len : int
        Maximum sequence length.
    dropout_rate : float
        Dropout rate for regularization.
    encoding_type : str
        Type of positional encoding, either "sinusoidal" or "learnable".
    hidden_ff_d : int
        Dimensionality of the feedforward network hidden layer.
    num_encoder_layers : int
        Number of encoder layers.
    num_decoder_layers : int
        Number of decoder layers.
    """

    def __init__(self, vocab_size, d_model, n_heads, max_len, dropout_rate, encoding_type, hidden_ff_d,
                 num_encoder_layers, num_decoder_layers):
        """
        Full Transformer model combining encoder and decoder components.

        This follows the architecture described in the paper "Attention is All You Need".

        Parameters
        ----------
        vocab_size : int
            Vocabulary size shared across encoder and decoder.
        d_model : int
            Dimensionality of embeddings and hidden layers.
        n_heads : int
            Number of attention heads.
        max_len : int
            Maximum sequence length.
        dropout_rate : float
            Dropout rate for regularization.
        encoding_type : str
            Type of positional encoding, either "sinusoidal" or "learnable".
        hidden_ff_d : int
            Dimensionality of the feedforward network hidden layer.
        num_encoder_layers : int
            Number of encoder layers.
        num_decoder_layers : int
            Number of decoder layers.
        """
        super().__init__()

        # Encoder stack
        self.encoder = TransformerEncoderModel(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, max_len=max_len,
            dropout_rate=dropout_rate, encoding_type=encoding_type, hidden_ff_d=hidden_ff_d,
            num_encoder_layers=num_encoder_layers
        )

        # Decoder stack
        self.decoder = TransformerDecoderModel(
            vocab_size=vocab_size, d_model=d_model, max_len=max_len,
            dropout_rate=dropout_rate, encoding_type=encoding_type,
            n_heads=n_heads, hidden_ff_d=hidden_ff_d, num_decoder_layers=num_decoder_layers
        )

        # Final projection to map decoder output to vocab size
        self.vocab_projection = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, past_key_values=None):
        """
        Forward pass through the full Transformer model.

        Parameters
        ----------
        src : Tensor
            Source token ids of shape (batch_size, src_len).
        tgt : Tensor
            Target token ids of shape (batch_size, tgt_len).
        src_mask : Tensor, optional
            Source padding mask of shape (batch_size, 1, 1, src_len).
        tgt_mask : Tensor, optional
            Target look ahead mask of shape (batch_size, 1, tgt_len, tgt_len).
        past_key_values : list of tuple, optional
            Cached key and value states for incremental decoding.

        Returns
        -------
        Tensor
            Logits over the vocabulary of shape (batch_size, tgt_len, vocab_size).
        """
        encoder_output = self.encode(src=src, src_mask=src_mask)
        decoder_output, _ = self.decode(
            tgt=tgt, encoder_output=encoder_output,
            tgt_mask=tgt_mask, src_mask=src_mask,
            past_key_values=past_key_values
        )
        vocab_logits = self.vocab_projection(decoder_output)
        return vocab_logits

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence into hidden representations.

        Parameters
        ----------
        src : Tensor
            Source token ids of shape (batch_size, src_len).
        src_mask : Tensor, optional
            Padding mask for the source.

        Returns
        -------
        Tensor
            Encoder output of shape (batch_size, src_len, d_model).
        """
        return self.encoder(x=src, mask=src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None,
               past_key_values=None, return_attention=False):
        """
        Decode the target sequence conditioned on the encoder output.

        Parameters
        ----------
        tgt : Tensor
            Target input token ids of shape (batch_size, tgt_len).
        encoder_output : Tensor
            Encoder hidden states.
        src_mask : Tensor, optional
            Source mask.
        tgt_mask : Tensor, optional
            Target mask.
        past_key_values : list of tuple, optional
            Cached key and value pairs for incremental decoding.
        return_attention : bool
            If True, also returns cross attention weights.

        Returns
        -------
        tuple
            If return_attention is False:
                (decoder_output, updated_past_key_values)
            If return_attention is True:
                (decoder_output, updated_past_key_values, cross_attention_weights)
        """
        if return_attention:
            x, updated_past, cross_attn = self.decoder(
                x=tgt, encoder_output=encoder_output,
                tgt_mask=tgt_mask, src_mask=src_mask,
                past_key_values=past_key_values,
                return_attention=True
            )
            return x, updated_past, cross_attn
        else:
            x, updated_past = self.decoder(
                x=tgt, encoder_output=encoder_output,
                tgt_mask=tgt_mask, src_mask=src_mask,
                past_key_values=past_key_values
            )
            return x, updated_past
