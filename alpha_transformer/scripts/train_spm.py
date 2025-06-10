import sentencepiece as spm
import os
import argparse

# Estimate vocab size based on number of sentences
def estimate_vocab_size(file_path):
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            line_count += 1

    # 2 lines per sentence pair (src + tgt), so divide by 2
    num_sentences = line_count // 2

    # Simple heuristic
    if num_sentences < 100_000:
        return 8000
    elif num_sentences < 500_000:
        return 16000
    elif num_sentences < 1_000_000:
        return 32000
    else:
        return 50000
    
def train_sentencepiece(input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,  # 'bpe' or 'unigram'
        character_coverage=1.0, # Good for english, french, german
        bos_id=1,                  # <sos>
        eos_id=2,               # <eos>
        pad_id=3                # <pad>
    )

    print(f"Trained SentencePiece model: {model_prefix}.model and {model_prefix}.vocab")


if __name__ == '__main__':
    # arg parser for needed parameters
    parser = argparse.ArgumentParser(description="Train SentencePiece model for MT")
    parser.add_argument('--lang', type=str, required=True, help='Language pair (e.g., en-fr or en-de)')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'], help='Model type: bpe or unigram')
    parser.add_argument('--all_vocab_sizes', action='store_true', help='Train models with 8k, 16k, 32k vocab sizes')

    args = parser.parse_args()
    data_folder = 'data'
    if args.lang == 'en-fr':
        input_file = os.path.join(data_folder, 'train_en_fr.all')
        base_model_prefix = os.path.join(data_folder, 'spm_en_fr') # saves as data/spm.model, data/spm.vocab
    elif args.lang == 'en-de':
        input_file = os.path.join(data_folder, 'train_en_de.all')
        base_model_prefix = os.path.join(data_folder, 'spm_en_de')
    else:
        raise ValueError(f"Unsupported language pair: {args.lang}")
    
    # train for multiple vocab sizes
    if args.all_vocab_sizes:
        vocab_sizes = [8000, 16000, 32000]
        for vocab_size in vocab_sizes:
            model_prefix = f'{base_model_prefix}_{vocab_size//1000}k'
            train_sentencepiece(input_file=input_file, model_prefix=model_prefix,
                                vocab_size=vocab_size)
    else:
        if args.vocab_size is None:
            vocab_size = estimate_vocab_size(input_file)
            print(f"Auto-detected vocab size: {vocab_size}")
        else:
            vocab_size = args.vocab_size
        model_prefix=base_model_prefix
        # Good size for local testing: 800
        train_sentencepiece(input_file, model_prefix=model_prefix, vocab_size=args.vocab_size, model_type=args.model_type)