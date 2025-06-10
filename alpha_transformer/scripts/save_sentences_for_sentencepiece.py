from datasets import load_dataset

def save_french_sentences_for_spm(out_file='data/train_en_fr.all'):
    # Load French dataset
    dataset = load_dataset('iwslt2017', 'iwslt2017-en-fr', trust_remote_code=True)
    train_data = dataset['train']

    with open(out_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            src_text = item['translation']['en'].strip()
            tgt_text = item['translation']['fr'].strip()
            f.write(src_text + '\n')
            f.write(tgt_text + '\n')
    print(f'Saved {out_file} for SentencePiece training.')

def save_german_sentences_for_spm(out_file='data/train_en_de.all'):
    # Load French dataset
    dataset = load_dataset('wmt16', 'de-en', trust_remote_code=True)
    train_data = dataset['train']

    with open(out_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            src_text = item['translation']['en'].strip()
            tgt_text = item['translation']['de'].strip()
            f.write(src_text + '\n')
            f.write(tgt_text + '\n')
    print(f'Saved {out_file} for SentencePiece training.')

if __name__ == "__main__":
    # Load dataset
    # Save sentences
    save_french_sentences_for_spm(out_file='data/train_en_fr.all')
    save_german_sentences_for_spm(out_file='data/train_en_de.all')
