import torch.nn as nn
from transformer.core.input_embedding import InputEmbedding
from transformer.decoder.transformer_decoder import TransformerDecoder

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
    
    # x shape: (batch_size, tgt_len) - tgt decoder input
    # encoder_output: (batch_size, src_len, d_model)
    # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
    # src_mask: (batch_size, 1, 1, src_len)
    # output: (batch_size, tget_len, d_model)
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None,
                past_key_values=None, return_attention=False):
        x = self.input_embedding(x)
        if return_attention:
            x, updated_past_key_values, weighted_attention_matrix = self.decoder_stack(x, encoder_output, tgt_mask=tgt_mask,
                                src_mask=src_mask,past_key_values=past_key_values, return_attention=return_attention)
            return x, updated_past_key_values, weighted_attention_matrix
        else:
            x, updated_past_key_values = self.decoder_stack(x, encoder_output, tgt_mask=tgt_mask,
                                src_mask=src_mask,past_key_values=past_key_values)
            return x, updated_past_key_values