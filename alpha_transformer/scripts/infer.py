import torch
import argparse
from trainer import Trainer
from transformer.transformer import Transformer
import sentencepiece as spm
import os
import json

# ================================
# Load model checkpoint and tokenizer
# ================================
def load_checkpoint_and_tokenizer(checkpoint_path):
    """
    Loads a Transformer model checkpoint along with its tokenizer.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint

    Returns:
        trainer (Trainer): Trainer object with model loaded
    """
    # Load checkpoint to CPU to avoid device mismatch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = argparse.Namespace(**checkpoint["args"])  # Reconstruct arguments

    # Load SentencePiece tokenizer
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load(args.sp_model_path)
    vocab_size = sp_tokenizer.get_piece_size()
    # Instantiate model with architecture from saved args
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_decoder_layers=args.num_layers,
        num_encoder_layers=args.num_layers,
        max_len=args.max_len,
        dropout_rate=args.dropout_rate,
        hidden_ff_d=args.d_model * 4,
        encoding_type=args.encoding_type,
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Wrap in Trainer and load full checkpoint (weights + stats)
    trainer = Trainer(model=model, tokenizer=sp_tokenizer)
    trainer.load_checkpoint(checkpoint_path)

    return trainer

# =========================
# Non-batched inference
# =========================
def translate_sentences_non_batched(trainer, sentences, decode_type, beam_size, return_attention=False):
    """
    Translates a list of sentences one at a time.

    Args:
        trainer (Trainer): Loaded trainer with model
        sentences (List[str]): Sentences to translate
        decode_type (str): 'beam' or 'greedy'
        beam_size (int): Beam width for decoding
        return_attention (bool): Whether to return attention weights

    Returns:
        Tuple: (decoded sentences, input tokens, output tokens, attentions)
    """
    results, input_tokens, output_tokens, attentions = [], [], [], []

    for sentence in sentences:
        token_ids = trainer.tokenizer.encode(sentence, out_type=int)
        input_tokens.append([trainer.tokenizer.IdToPiece(id) for id in token_ids])

        input_tensor = torch.tensor(token_ids).unsqueeze(0).to(trainer.device)  # Add batch dim

        if return_attention:
            decoded_sentence, output_ids, cross_attention = trainer.infer(
                input_tensor, decode_type=decode_type, beam_size=beam_size, return_attention=True)
            attentions.append(cross_attention)
        else:
            decoded_sentence, output_ids = trainer.infer(
                input_tensor, decode_type=decode_type, beam_size=beam_size)

        if isinstance(output_ids[0], list):  # Handle beam output
            output_ids = output_ids[0]

        output_tokens.append([trainer.tokenizer.IdToPiece(id) for id in output_ids])
        results.append(decoded_sentence)

    return results, input_tokens, output_tokens, attentions

# =========================
# Batched inference
# =========================
def translate_sentences(trainer, sentences, decode_type, beam_size, return_attention=False):
    """
    Translates a batch of sentences with padding.

    Args:
        trainer (Trainer): Loaded trainer
        sentences (List[str]): Input text
        decode_type (str): 'beam' or 'greedy'
        beam_size (int): Beam width
        return_attention (bool): Return attention weights if True

    Returns:
        Tuple: (translations, input_tokens, output_tokens, attentions)
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    token_batches = [trainer.tokenizer.encode(s, out_type=int) for s in sentences]
    input_tokens = [[trainer.tokenizer.IdToPiece(id) for id in seq] for seq in token_batches]

    # Pad input sequences to uniform length
    input_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in token_batches],
        batch_first=True,
        padding_value=trainer.special_tokens['<pad>']
    ).to(trainer.device)

    # Perform inference
    if return_attention:
        results, output_ids, attentions = trainer.infer(
            input_tensor, decode_type=decode_type, beam_size=beam_size, return_attention=True)
    else:
        results, output_ids = trainer.infer(
            input_tensor, decode_type=decode_type, beam_size=beam_size)
        attentions = None

    output_tokens = [[trainer.tokenizer.IdToPiece(id) for id in seq] for seq in output_ids]
    return results, input_tokens, output_tokens, attentions

# ================================
# Main CLI utility
# ================================
def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained Transformer model.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to best_model.pt checkpoint')
    parser.add_argument('--text', type=str, help='Translate a single sentence')
    parser.add_argument('--input_file', type=str, help='Path to input file with one sentence per line')
    parser.add_argument('--output_file', type=str, help='Optional output file to write translations')
    parser.add_argument('--decode_type', type=str, default='beam', choices=['beam', 'greedy'],
                        help='Decoding strategy')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam width for beam search')
    args = parser.parse_args()

    # Validate input source
    if not args.text and not args.input_file:
        raise ValueError("You must specify either --text or --input_file")

    # Load model and tokenizer
    trainer = load_checkpoint_and_tokenizer(args.checkpoint)

    # Load text from file or argument
    if args.text:
        sentences = [args.text]
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

    # Translate
    translations, _, _, _ = translate_sentences(
        trainer=trainer,
        sentences=sentences,
        decode_type=args.decode_type,
        beam_size=args.beam_size
    )

    # Output to file or print
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for line in translations:
                f.write(line + '\n')
    else:
        for sentence, translation in zip(sentences, translations):
            print(f"Input: {sentence}")
            print(f"Output: {translation}\n")

# Entry point
if __name__ == '__main__':
    main()
