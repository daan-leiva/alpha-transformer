import torch
import torch.nn as nn
from encoder.encoder_model import TransformerEncoderModel
from decoder.decoder_model import TransformerDecoderModel

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, max_len, dropout_rate, encoding_type, hidden_ff_d,
                 num_encoder_layers, num_decoder_layers):
        super().__init__()
        # complete encoder
        self.encoder = TransformerEncoderModel(vocab_size=vocab_size, d_model=d_model,
                                               n_heads=n_heads, max_len=max_len,
                                               dropout_rate=dropout_rate, encoding_type=encoding_type,
                                               hidden_ff_d=hidden_ff_d,
                                               num_encoder_layers=num_encoder_layers)
        # complete decoder
        self.decoder = TransformerDecoderModel(vocab_size=vocab_size, d_model=d_model, max_len=max_len,
                                               dropout_rate=dropout_rate, encoding_type=encoding_type,
                                               n_heads=n_heads, hidden_ff_d=hidden_ff_d,
                                               num_decoder_layers=num_decoder_layers)
        # convers output from encoder/decoder into a vocab again
        self.vocab_projection = nn.Linear(d_model, vocab_size)


    def forward(self, src, target, src_mask, target_mask):
        # src shape: (batch_size, src_len)
        # target: (batch_size, target_len)
        # src_mask: (batch_size, 1, 1, src_len)
        # target_mask: (batch_size, 1, target_len, target_len)
        encoder_output = self.encoder(src, src_mask=src_mask)
        decoder_output = self.decoder(target, encoder_output, target_mask, src_mask)

        # logits to return vocab instead of a mode
        vocab_logits = self.vocab_projection(decoder_output)

        return vocab_logits