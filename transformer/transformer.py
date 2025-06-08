import torch.nn as nn
from transformer.encoder.encoder_model import TransformerEncoderModel
from transformer.decoder.decoder_model import TransformerDecoderModel

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


    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                past_key_values=None):
        # src shape: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        # src_mask: (batch_size, 1, 1, src_len)
        # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        encoder_output = self.encode(src=src, src_mask=src_mask)
        decoder_output, _ = self.decode(tgt=tgt, encoder_output=encoder_output,
                                      tgt_mask=tgt_mask, src_mask=src_mask,
                                      past_key_values=past_key_values)
        
        # logits to return vocab instead of a mode
        vocab_logits = self.vocab_projection(decoder_output)

        return vocab_logits
    
    def encode(self, src, src_mask=None):
        # src shape: (batch_size, src_len)
        # src_mask: (batch_size, 1, 1, src_len)
        encoder_output = self.encoder(x=src, mask=src_mask)
        
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None,
               past_key_values=None):
        # tgt: (batch_size, tgt_len)
        # tgt_mask: (batch_size, 1, tgt_len, tgt_len)
        decoder_output, updated_past_key_values = self.decoder(
                            x=tgt, encoder_output=encoder_output,
                            tgt_mask=tgt_mask, src_mask=src_mask,
                            past_key_values=past_key_values)
         
        return decoder_output, updated_past_key_values