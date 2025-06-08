from datasets import load_dataset

def save_sentences_for_spm(dataset, src_lang='en', tgt_lang='fr', out_file='data/train.all'):
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            src_text = item['translation'][src_lang].strip()
            tgt_text = item['translation'][tgt_lang].strip()
            f.write(src_text + '\n')
            f.write(tgt_text + '\n')
    print(f'Saved {out_file} for SentencePiece training.')

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset('iwslt2017', 'iwslt2017-en-fr', trust_remote_code=True)
    train_data = dataset['train']

    # Save sentences
    save_sentences_for_spm(train_data, src_lang='en', tgt_lang='fr', out_file='data/train.all')