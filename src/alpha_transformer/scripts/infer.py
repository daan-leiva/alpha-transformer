import torch
import argparse
from alpha_transformer.trainer import Trainer
from alpha_transformer.transformer.transformer import Transformer
import sentencepiece as spm


def load_checkpoint_and_tokenizer(checkpoint_path):
    """
    Load a Transformer checkpoint and its associated tokenizer.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved best_model.pt file.

    Returns
    -------
    Trainer
        Trainer instance with model weights loaded and tokenizer attached.
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


def translate_sentences_non_batched(trainer, sentences, decode_type, beam_size, return_attention=False):
    """
    Translate sentences one by one without padding.

    This path is useful if you want simple per sentence behavior and more
    detailed control per example.

    Parameters
    ----------
    trainer : Trainer
        Trainer with a loaded model and tokenizer.
    sentences : list[str]
        Sentences to translate.
    decode_type : str
        "beam" or "greedy".
    beam_size : int
        Beam width used when decode_type is "beam".
    return_attention : bool
        If True, returns cross attention tensors for each sentence.

    Returns
    -------
    tuple
        (decoded_sentences, input_tokens, output_tokens, attentions)
    """
    results, input_tokens, output_tokens, attentions = [], [], [], []

    for sentence in sentences:
        token_ids = trainer.tokenizer.encode(sentence, out_type=int)
        input_tokens.append([trainer.tokenizer.IdToPiece(id) for id in token_ids])

        # Shape is (1, seq_len) to represent a single batch
        input_tensor = torch.tensor(token_ids).unsqueeze(0).to(trainer.device)  # Add batch dim

        if return_attention:
            decoded_sentence, output_ids, cross_attention = trainer.infer(
                input_tensor, decode_type=decode_type, beam_size=beam_size, return_attention=True)
            attentions.append(cross_attention)
        else:
            decoded_sentence, output_ids = trainer.infer(
                input_tensor, decode_type=decode_type, beam_size=beam_size)

        # If beam search returns a list of beams, keep only the top beam
        if isinstance(output_ids[0], list):  # Handle beam output
            output_ids = output_ids[0]

        output_tokens.append([trainer.tokenizer.IdToPiece(id) for id in output_ids])
        results.append(decoded_sentence)

    return results, input_tokens, output_tokens, attentions


def translate_sentences(trainer, sentences, decode_type, beam_size, return_attention=False):
    """
    Translate a batch of sentences with padding.

    Parameters
    ----------
    trainer : Trainer
        Trainer instance with model and tokenizer.
    sentences : list[str] or str
        Input sentences. A single string will be wrapped in a list.
    decode_type : str
        "beam" or "greedy".
    beam_size : int
        Beam width for beam search.
    return_attention : bool
        If True, also return cross attention tensors.

    Returns
    -------
    tuple
        (translations, input_tokens, output_tokens, attentions)
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    token_batches = [trainer.tokenizer.encode(s, out_type=int) for s in sentences]
    input_tokens = [[trainer.tokenizer.IdToPiece(id) for id in seq] for seq in token_batches]

    # Pad sequences so they can be processed as a single batch
    input_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in token_batches],
        batch_first=True,
        padding_value=trainer.special_tokens['<pad>']
    ).to(trainer.device)

    if return_attention:
        results, output_ids, attentions = trainer.infer(
            input_tensor, decode_type=decode_type, beam_size=beam_size, return_attention=True)
    else:
        results, output_ids = trainer.infer(
            input_tensor, decode_type=decode_type, beam_size=beam_size)
        attentions = None

    output_tokens = [[trainer.tokenizer.IdToPiece(id) for id in seq] for seq in output_ids]

    return results, input_tokens, output_tokens, attentions


def main():
    """
    Command line translation helper.

    Example usages:
        python infer.py --checkpoint path/to/best_model.pt --text "Hello"
        python infer.py --checkpoint path/to/best_model.pt --input_file input.txt
                        --output_file output.txt
    """
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


if __name__ == '__main__':
    main()
