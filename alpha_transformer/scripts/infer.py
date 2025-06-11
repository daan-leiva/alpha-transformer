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
                         encoding_type='sinusoidal'
                         ).to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # create an empty Trainer
    trainer = Trainer(model=model, tokenizer=sp_tokenizer)
    # load trainer values (effectively alos loading model state dict)
    # for inference so optimzer and scheduler left as false
    trainer.load_checkpoint(checkpoint_path)

    return trainer

def translate_sentences(trainer, sentences, decode_type, beam_size):
    results = []
    for sentence in sentences:
        token_ids = trainer.tokenizer.encode(sentence, out_type=int)
        input_tensor = torch.tensor(token_ids).unsqueeze(0)  # batch size 1
        # move to same device
        input_tensor = input_tensor.to(trainer.device)
        decoded_sentence = trainer.infer(input_tensor, type = decode_type, beam_size=beam_size)
        results.append(decoded_sentence)
    return results

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

    translations = translate_sentences(trainer=trainer, sentences=sentences,
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