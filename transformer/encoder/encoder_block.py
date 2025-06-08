import torch.nn as nn
from transformer.core.multihead_attention import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    # d_ff = dimension of the feedforward network's hidden layer
    def __init__(self, d_model, n_heads, hidden_ff_d, dropout_rate=0.1):
        super().__init__()

        # multi head attention layer
        self.multi_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout_rate=dropout_rate)
        # layer normalization layer (also used to merge residual)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        # feedforward network to introduce non-linearity
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_ff_d),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_ff_d, d_model)
        )

        # normalization layer (also used as a residual layer)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        # dropout layer
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        self_attention_output, _, _ = self.multi_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(self_attention_output))
        
        # feed forward + residual connection
        ff_input = x
        x = self.ff(x)
        x = self.norm2(ff_input + self.dropout2(x))

        return x