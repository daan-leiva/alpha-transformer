import torch.nn as nn
import torch
from transformer.decoder.decoder_block import TransformerDecoderBlock

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d, num_decoder_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderBlock(d_model, n_heads,
                                                             dropout_rate, hidden_ff_d)
                                     for _ in range(num_decoder_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None,
                past_key_values=None, return_attention=False):
        updated_key_values = []
        weighted_attention_matrix = []
        for i, layer in enumerate(self.layers):
            past_key_value = None
            if past_key_values is not None:
                past_key_value = past_key_values[i]
            if return_attention:
                x, updated_key_value, weighted_attention_matrix_single = layer(x, encoder_output, tgt_mask=tgt_mask,
                      src_mask=src_mask, past_key_value=past_key_value, return_attention=return_attention)
                weighted_attention_matrix.append(weighted_attention_matrix_single)
            else:
                x, updated_key_value = layer(x, encoder_output, tgt_mask=tgt_mask,
                      src_mask=src_mask, past_key_value=past_key_value, return_attention=return_attention)
            updated_key_values.append(updated_key_value)
        x = self.norm(x)
        if return_attention:
            weighted_attention_matrix = torch.stack(weighted_attention_matrix, dim=0)
            return x, updated_key_values, weighted_attention_matrix
        else:
            return x, updated_key_values