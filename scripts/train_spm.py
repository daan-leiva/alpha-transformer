import sentencepiece as spm
import os

def train_sentencepiece(input_file, model_prefix, vocab_size=32000, model_type='bpe'):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,  # 'bpe' or 'unigram'
        character_coverage=1.0, # Good for english and french
        bos_id=1,                  # <sos>
        eos_id=2,               # <eos>
        pad_id=3                # <pad>
    )

    print(f"Trained SentencePiece model: {model_prefix}.model and {model_prefix}.vocab")


if __name__ == '__main__':
    data_folder = 'data'
    input_file = os.path.join(data_folder, 'train.all')
    model_prefix = os.path.join(data_folder, 'spm') # saves as data/spm.model, data/spm.vocab

    # Good size for local testing: 800
    train_sentencepiece(input_file, model_prefix=model_prefix, vocab_size=8000, model_type='bpe')