from datasets import load_dataset

# ==========================================
# Save English–French parallel corpus for SPM
# ==========================================
def save_french_sentences_for_spm(out_file='data/train_en_fr.all'):
    # Load the IWSLT 2017 English–French translation dataset
    dataset = load_dataset('iwslt2017', 'iwslt2017-en-fr', trust_remote_code=True)
    train_data = dataset['train']

    # Write both source and target sentences (EN & FR) line-by-line into a single file
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            src_text = item['translation']['en'].strip()  # Source: English
            tgt_text = item['translation']['fr'].strip()  # Target: French
            f.write(src_text + '\n')
            f.write(tgt_text + '\n')

    print(f'Saved {out_file} for SentencePiece training.')

# ==========================================
# Save English–German parallel corpus for SPM
# ==========================================
def save_german_sentences_for_spm(out_file='data/train_en_de.all'):
    # Load the WMT16 English–German translation dataset
    dataset = load_dataset('wmt16', 'de-en', trust_remote_code=True)
    train_data = dataset['train']

    # Write both source and target sentences (EN & DE) line-by-line into a single file
    with open(out_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            src_text = item['translation']['en'].strip()  # Source: English
            tgt_text = item['translation']['de'].strip()  # Target: German
            f.write(src_text + '\n')
            f.write(tgt_text + '\n')

    print(f'Saved {out_file} for SentencePiece training.')

# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    # Generate data for both EN–FR and EN–DE
    save_french_sentences_for_spm(out_file='data/train_en_fr.all')
    save_german_sentences_for_spm(out_file='data/train_en_de.all')
