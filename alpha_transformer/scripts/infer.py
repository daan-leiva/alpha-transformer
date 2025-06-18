import torch
import argparse
from trainer import Trainer
from transformer.transformer import Transformer
import sentencepiece as spm
import os
import json

# loads the trainer, model and sp_tokenizer
def load_checkpoint_and_tokenizer(checkpoint_path):
    # load the checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = argparse.Namespace(**checkpoint["args"])

    # Load tokenizer
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load(args.sp_model_path)
    vocab_size = sp_tokenizer.get_piece_size()

    # Rebuild model exactly as in training
    model = Transformer(vocab_size=vocab_size,
                         d_model=args.d_model,
                         n_heads=args.n_heads,
                         num_decoder_layers=args.num_layers,
                         num_encoder_layers=args.num_layers,
                         max_len=args.max_len,
                         dropout_rate=args.dropout_rate,
                         hidden_ff_d=args.d_model*4,
                         encoding_type=args.encoding_type,
                         ).to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # create an empty Trainer
    trainer = Trainer(model=model, tokenizer=sp_tokenizer)
    # load trainer values (effectively alos loading model state dict)
    # for inference so optimzer and scheduler left as false
    trainer.load_checkpoint(checkpoint_path)

    return trainer

def translate_sentences_non_batched(trainer, sentences, decode_type, beam_size, return_attention=False):
    results = []
    input_tokens = []
    output_tokens = []
    attentions = []
    for sentence in sentences:
        token_ids = trainer.tokenizer.encode(sentence, out_type=int)
        input_tokens.append([trainer.tokenizer.IdToPiece(id) for id in token_ids])
        input_tensor = torch.tensor(token_ids).unsqueeze(0)  # batch size 1
        # move to same device
        input_tensor = input_tensor.to(trainer.device)
        if return_attention:
            decoded_sentence, output_ids, cross_attention = trainer.infer(input_tensor, decode_type = decode_type,
                                                     beam_size=beam_size, return_attention=return_attention)
            attentions.append(cross_attention)
        else:
            decoded_sentence, output_ids = trainer.infer(input_tensor, decode_type = decode_type,
                                                     beam_size=beam_size)
        # check if the returned type is a list of lists
        if isinstance(output_ids[0], list):
            output_ids = output_ids[0]
        output_tokens.append([trainer.tokenizer.IdToPiece(id) for id in output_ids])
        results.append(decoded_sentence)
    return results, input_tokens, output_tokens, attentions


def translate_sentences(trainer, sentences, decode_type, beam_size, return_attention=False):
     # Ensure sentences is a list
    if isinstance(sentences, str):
        sentences = [sentences]

    # Tokenize all sentences
    token_batches = [trainer.tokenizer.encode(s, out_type=int) for s in sentences]
    input_tokens = [[trainer.tokenizer.IdToPiece(id) for id in seq] for seq in token_batches]

    # Convert to tensor
    input_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in token_batches],
        batch_first=True,
        padding_value=trainer.special_tokens['<pad>']
    ).to(trainer.device)

    # Run inference on the batch
    if return_attention:
        results, output_ids, attentions = trainer.infer(
            input_tensor, decode_type=decode_type, beam_size=beam_size, return_attention=True
        )
    else:
        results, output_ids = trainer.infer(
            input_tensor, decode_type=decode_type, beam_size=beam_size, return_attention=False
        )
        attentions = None

    # Decode output token IDs to strings
    output_tokens = [[trainer.tokenizer.IdToPiece(id) for id in seq] for seq in output_ids]

    return results, input_tokens, output_tokens, attentions


def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained Transformer model.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the best_model.pt checkpoint')
    parser.add_argument('--text', type=str, help='Translate a single sentence')
    parser.add_argument('--input_file', type=str, help='File with one sentence per line')
    parser.add_argument('--output_file', type=str, help='Optional: file to write translations to')
    parser.add_argument('--decode_type', type=str, default='beam', choices=['beam', 'greedy'],
                        help='Decoding method to use')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    
    args = parser.parse_args()

    if not args.text and not args.input_file:
        raise ValueError("You must specify either --text or --input_file")
    
    trainer = load_checkpoint_and_tokenizer(args.checkpoint)

    if args.text:
        sentences = [args.text]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

    translations, _, _, _ = translate_sentences(trainer=trainer, sentences=sentences,
                                       decode_type=args.decode_type,
                                       beam_size=args.beam_size)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for line in translations:
                f.write(line + '\n')
    else:
        for sentence, translation in zip(sentences, translations):
            print(f"Input: {sentence}")
            print(f"Output: {translation}\n")

if __name__ == '__main__':
    main()