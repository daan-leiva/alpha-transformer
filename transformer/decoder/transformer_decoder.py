import torch.nn as nn
from decoder.decoder_block import TransformerDecoderBlock

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate, hidden_ff_d, num_decoder_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderBlock(d_model, n_heads,
                                                             dropout_rate, hidden_ff_d)
                                     for _ in range(num_decoder_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, target_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask=target_mask, src_mask=src_mask)
        x = self.norm(x)
        return x