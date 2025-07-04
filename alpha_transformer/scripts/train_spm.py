import sentencepiece as spm
import os
import argparse

# === Estimate vocabulary size based on file line count ===
def estimate_vocab_size(file_path):
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            line_count += 1

    # Assume each sentence pair spans two lines (source + target)
    num_sentences = line_count // 2

    # Heuristic to choose vocab size
    if num_sentences < 100_000:
        return 8000
    elif num_sentences < 500_000:
        return 16000
    elif num_sentences < 1_000_000:
        return 32000
    else:
        return 50000

# === Train a SentencePiece tokenizer ===
def train_sentencepiece(input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,          # Options: 'bpe', 'unigram'
        character_coverage=1.0,         # Full coverage, good for Latin alphabets
        bos_id=1,                       # Assign <sos> token ID = 1
        eos_id=2,                       # Assign <eos> token ID = 2
        pad_id=3                        # Assign <pad> token ID = 3
    )
    print(f"Trained SentencePiece model: {model_prefix}.model and {model_prefix}.vocab")

# === Main CLI entry point ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SentencePiece model for machine translation")
    parser.add_argument('--lang', type=str, required=True,
                        help='Language pair (e.g., en-fr or en-de)')
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='Vocabulary size (overridden if --all_vocab_sizes is used)')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'],
                        help='Model type for SentencePiece: bpe or unigram')
    parser.add_argument('--all_vocab_sizes', action='store_true',
                        help='Train models with 8k, 16k, and 32k vocab sizes instead of one')

    args = parser.parse_args()

    # === Choose input file and output prefix based on language pair ===
    data_folder = 'data'
    if args.lang == 'en-fr':
        input_file = os.path.join(data_folder, 'train_en_fr.all')
        base_model_prefix = os.path.join(data_folder, 'spm_en_fr')
    elif args.lang == 'en-de':
        input_file = os.path.join(data_folder, 'train_en_de.all')
        base_model_prefix = os.path.join(data_folder, 'spm_en_de')
    else:
        raise ValueError(f"Unsupported language pair: {args.lang}")

    # === Train for multiple vocab sizes if specified ===
    if args.all_vocab_sizes:
        vocab_sizes = [8000, 16000, 32000]
        for vocab_size in vocab_sizes:
            model_prefix = f'{base_model_prefix}_{vocab_size // 1000}k'
            train_sentencepiece(input_file=input_file,
                                model_prefix=model_prefix,
                                vocab_size=vocab_size)
    else:
        # Use provided or auto-estimated vocab size
        if args.vocab_size is None:
            vocab_size = estimate_vocab_size(input_file)
            print(f"Auto-detected vocab size: {vocab_size}")
        else:
            vocab_size = args.vocab_size

        model_prefix = base_model_prefix
        train_sentencepiece(input_file=input_file,
                            model_prefix=model_prefix,
                            vocab_size=vocab_size,
                            model_type=args.model_type)
