import torch.nn as nn
import torch
from transformer.decoder.decoder_block import TransformerDecoderBlock

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder stack consisting of multiple TransformerDecoderBlocks.

    Args:
        d_model (int): Model embedding size
        n_heads (int): Number of attention heads
        dropout_rate (float): Dropout probability
        hidden_ff_d (int): Hidden dimension size in feedforward layers
        num_decoder_layers (int): Number of stacked decoder blocks
    """
    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d, num_decoder_layers):
        super().__init__()

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, dropout_rate, hidden_ff_d)
            for _ in range(num_decoder_layers)
        ])

        # Final layer normalization (applied after all decoder layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None,
                past_key_values=None, return_attention=False):
        """
        Forward pass through the Transformer decoder stack.

        Args:
            x (Tensor): Input tensor (batch_size, tgt_len, d_model)
            encoder_output (Tensor): Output from encoder (batch_size, src_len, d_model)
            tgt_mask (Tensor): Mask for decoder self-attention (batch_size, 1, tgt_len, tgt_len)
            src_mask (Tensor): Mask for encoder-decoder attention (batch_size, 1, 1, src_len)
            past_key_values (List[Tuple[Tensor, Tensor]]): Cached key/value pairs per layer
            return_attention (bool): If True, return attention weights for visualization

        Returns:
            x (Tensor): Output tensor (batch_size, tgt_len, d_model)
            updated_key_values (List[Tuple]): Updated key/value cache from each decoder block
            weighted_attention_matrix (Tensor, optional): Stack of attention matrices (num_layers, batch, heads, tgt_len, src_len)
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
