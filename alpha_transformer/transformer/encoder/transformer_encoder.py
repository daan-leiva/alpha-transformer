import torch.nn as nn
from transformer.encoder.encoder_block import TransformerEncoderBlock

class TransformerEncoder(nn.Module):
    """
    Stack of encoder blocks followed by a final layer normalization.

    This module expects input already projected into model space
    (batch_size, seq_len, d_model) and applies the same mask to each
    encoder block.
    """
    def __init__(self, d_model, n_heads, hidden_ff_d, num_encoder_layers, dropout_rate):
        """
        Parameters
        ----------
        d_model : int
            Embedding and hidden dimension.
        n_heads : int
            Number of attention heads in each encoder block.
        hidden_ff_d : int
            Hidden dimension of the feedforward sublayers.
        num_encoder_layers : int
            Number of encoder blocks in the stack.
        dropout_rate : float
            Dropout probability used in encoder blocks.
        """
        super().__init__()

        # Sequence of encoder blocks cannot use nn.Sequential because
        # each block needs both x and mask as inputs.
        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, hidden_ff_d, dropout_rate) for _ in range(num_encoder_layers)])

        # Final normalization over the model dimension
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder stack.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        mask : Tensor, optional
            Attention mask of shape (batch_size, 1, 1, seq_len). Same mask
            is used for all encoder layers.

        Returns
        -------
        Tensor
            Encoded representation of shape (batch_size, seq_len, d_model).
        """
        # Apply each encoder block in sequence
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Normalize once at the end
        x = self.norm(x)

        return x