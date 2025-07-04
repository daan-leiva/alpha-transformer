import torch.nn as nn
from transformer.core.input_embedding import InputEmbedding
from transformer.decoder.transformer_decoder import TransformerDecoder

class TransformerDecoderModel(nn.Module):
    """
    A Transformer decoder-only model that takes target input tokens and encoder output,
    and returns decoder hidden states. Used as part of an encoder-decoder Transformer.

    Components:
        - InputEmbedding: token + positional embedding
        - TransformerDecoder: stack of decoder layers (self-attention + cross-attention)
    """
    def __init__(self, vocab_size, d_model, max_len, dropout_rate,
                 encoding_type, n_heads, hidden_ff_d, num_decoder_layers):
        """
        Args:
            vocab_size (int): Size of vocabulary (number of unique tokens)
            d_model (int): Dimensionality of embeddings and model layers
            max_len (int): Maximum sequence length for positional encoding
            dropout_rate (float): Dropout rate for regularization
            encoding_type (str): 'sinusoidal' or 'learnable' positional encoding
            n_heads (int): Number of attention heads
            hidden_ff_d (int): Hidden dimension in feedforward sublayer
            num_decoder_layers (int): Number of stacked decoder blocks
        """
        super().__init__()

        # Input embedding combines token + position embeddings
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout_rate=dropout_rate,
            encoding_type=encoding_type
        )

        # Stack of TransformerDecoder layers
        self.decoder_stack = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            hidden_ff_d=hidden_ff_d,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None,
                past_key_values=None, return_attention=False):
        """
        Forward pass for Transformer decoder model.

        Args:
            x (Tensor): Target input tokens (batch_size, tgt_len)
            encoder_output (Tensor): Encoder output (batch_size, src_len, d_model)
            tgt_mask (Tensor): Decoder self-attention mask (batch_size, 1, tgt_len, tgt_len)
            src_mask (Tensor): Cross-attention mask (batch_size, 1, 1, src_len)
            past_key_values (List[Tuple]): Cached key/value pairs for each decoder layer (used in autoregressive decoding)
            return_attention (bool): Whether to return attention weights from the final decoder layer

        Returns:
            output (Tensor): Decoder output (batch_size, tgt_len, d_model)
            updated_past_key_values (List[Tuple]): Updated key/value pairs
            attention_weights (optional): Cross-attention weights from final decoder layer (if return_attention=True)
        """
        # Convert token IDs to embeddings + positional encodings
        x = self.input_embedding(x)

        # Decode with optional attention output
        if return_attention:
            x, updated_past_key_values, weighted_attention_matrix = self.decoder_stack(
                x, encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_mask,
                past_key_values=past_key_values,
                return_attention=True
            )
            return x, updated_past_key_values, weighted_attention_matrix
        else:
            x, updated_past_key_values = self.decoder_stack(
                x, encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_mask,
                past_key_values=past_key_values
            )
            return x, updated_past_key_values
