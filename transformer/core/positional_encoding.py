import torch.nn as nn
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):
    # takes input of shape (batch_size, seq_len, vocab_size)
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        positional_encoding = torch.zeros(max_len, d_model) # matrix to hold our output
        position = torch.arange(0, max_len).unsqueeze(1) # create a (max_len, 1) vector (the 1 dimension is to broadcast the multiplication)
        # rewrote the 1 / (10000 ** (i / d_model)) to avoid instability from large numbers
        # technically the inverse is used to make the design simpler
        # this has dimension of (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))

        # even positions are sine encoded
        # shapes (max_len, 1) and (d_model//2) generate a (max_len, d_model//2) matrix
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # odd positions are cosine encoded
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # add a leading dimension for batching
        positional_encoding = positional_encoding.unsqueeze(0)
        # save the tensor with the model
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        # shape of x : batch_size, seq_len
        seq_len = x.shape[1]
        # only use the portion of the encoding relevant to this input
        return x + self.positional_encoding[:, :seq_len, :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # instead of a vocab size we are using the posibble positions as the first dimension
        self.position_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # arange moves tensors to the cpu so enforce a move to the device in case this is being ran on a gpu
        # + repeat positions without creating a bigger memory footprint
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.position_embedding(positions)