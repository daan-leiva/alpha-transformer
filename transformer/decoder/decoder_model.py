import torch.nn as nn
from core.input_embedding import InputEmbedding
from decoder.transformer_decoder import TransformerDecoder

class  TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout_rate,
                 encoding_type, n_heads, hidden_ff_d, num_decoder_layers):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                               max_len=max_len,
                                               dropout_rate=dropout_rate,
                                               encoding_type=encoding_type)
        self.decoder_stack = TransformerDecoder(d_model=d_model,
                                                n_heads=n_heads, dropout_rate=dropout_rate, hidden_ff_d=hidden_ff_d, num_decoder_layers=num_decoder_layers)
    
    # x shape: (batch_size, target_len) - target decoder input
    # encoder_output: (batch_size, src_len, d_model)
    # target_mask: (batch_size, 1, target_len, target_len)
    # src_mask: (batch_size, 1, 1, src_len)
    # output: (batch_size, target_len, d_model)
    def forward(self, x, encoder_output, target_mask=None, src_mask=None):
        x = self.input_embedding(x)
        x = self.decoder_stack(x, encoder_output, target_mask, src_mask)
        return x