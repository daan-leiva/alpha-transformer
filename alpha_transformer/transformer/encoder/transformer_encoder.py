import torch.nn as nn
from transformer.encoder.encoder_block import TransformerEncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, hidden_ff_d, num_encoder_layers, dropout_rate):
        super().__init__()

        # create sequential layers for whole encoder
        # note: cannot use nn.Sequential due to needing two inputs (x and mask)
        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, hidden_ff_d, dropout_rate) for _ in range(num_encoder_layers)])

        # layer normalization
        self.norm = nn.LayerNorm(d_model)

    # input shape: (batch_size, seq_len, d_model)
    # output shape: (batch_size, seq_len, d_model)
    def forward(self, x, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x