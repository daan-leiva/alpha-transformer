import torch.nn as nn
import torch
from alpha_transformer.transformer.decoder.decoder_block import TransformerDecoderBlock


class TransformerDecoder(nn.Module):
    """
    Transformer decoder stack composed of multiple TransformerDecoderBlocks.

    The stack supports:
      masked self attention in each layer with cached keys and values
      cross attention over the encoder output
      optional return of cross attention weights per layer
    """

    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d, num_decoder_layers):
        """
        Parameters
        ----------
        d_model : int
            Model embedding size.
        n_heads : int
            Number of attention heads in each decoder block.
        dropout_rate : float
            Dropout probability.
        hidden_ff_d : int
            Hidden dimension size in feedforward layers.
        num_decoder_layers : int
            Number of stacked decoder blocks.
        """
        super().__init__()

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, dropout_rate, hidden_ff_d)
            for _ in range(num_decoder_layers)
        ])

        # Final layer normalization after all decoder layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None,
                past_key_values=None, return_attention=False):
        """
        Forward pass through the Transformer decoder stack.

        Parameters
        ----------
        x : Tensor
            Decoder input tensor of shape (batch_size, tgt_len, d_model).
        encoder_output : Tensor
            Encoder output tensor of shape (batch_size, src_len, d_model).
        tgt_mask : Tensor, optional
            Mask for decoder self attention of shape
            (batch_size, 1, tgt_len, tgt_len).
        src_mask : Tensor, optional
            Mask for encoder decoder attention of shape
            (batch_size, 1, 1, src_len).
        past_key_values : list of tuple(Tensor, Tensor), optional
            Cached key and value pairs per layer for self attention.
            Length of the list equals num_decoder_layers. Each tuple holds
            tensors of shape (batch_size, n_heads, past_len, d_k).
        return_attention : bool
            If True, returns cross attention weights for each layer.

        Returns
        -------
        x : Tensor
            Output tensor of shape (batch_size, tgt_len, d_model).
        updated_key_values : list of tuple(Tensor, Tensor)
            Updated key and value cache from each decoder block.
        weighted_attention_matrix : Tensor, optional
            Stack of cross attention matrices with shape
            (num_layers, batch_size, n_heads, tgt_len, src_len)
            if return_attention is True.
        """
        updated_key_values = []
        weighted_attention_matrix = []

        # Pass input through each decoder layer
        for i, layer in enumerate(self.layers):
            # Retrieve cached past key/values if available
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if return_attention:
                # Forward with attention outputs
                x, new_kv, attention_weights = layer(
                    x, encoder_output,
                    tgt_mask=tgt_mask,
                    src_mask=src_mask,
                    past_key_value=past_key_value,
                    return_attention=True
                )
                weighted_attention_matrix.append(attention_weights)
            else:
                # Forward without attention output
                x, new_kv = layer(
                    x, encoder_output,
                    tgt_mask=tgt_mask,
                    src_mask=src_mask,
                    past_key_value=past_key_value,
                    return_attention=False
                )

            updated_key_values.append(new_kv)

        # Apply final normalization
        x = self.norm(x)

        if return_attention:
            # Shape: (num_layers, batch_size, n_heads, tgt_len, src_len)
            weighted_attention_matrix = torch.stack(weighted_attention_matrix, dim=0)
            return x, updated_key_values, weighted_attention_matrix
        else:
            return x, updated_key_values
