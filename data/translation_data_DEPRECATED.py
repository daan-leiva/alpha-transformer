# THIS VERSION HAS BEEN DEPRECATED BY THE NEW VERSION THAT USES
# SENTENCE PIECING
import spacy
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# provides data for training
class TranslationDataset(Dataset):
    def __init__(self, data, src_lang, target_lang, src_tokenizer,
                    target_tokenizer, src_vocab, target_vocab,
                    max_len=100):
        super().__init__()
        # dictionary of data
        self.data = data
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_vocab = src_vocab
        self.target_vocab = target_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data) # same len as target sentences
    
    def encode(self, tokens, vocab):
        ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
        return ids[:self.max_len]
    
    # returns the src and target sentences tokenized and coverted to ids
    def __getitem__(self, idx):
        sample = self.data[idx]['translation']
        src_text = sample[self.src_lang]
        target_text = sample[self.target_lang]
        src_tokens = ['<sos>'] + self.src_tokenizer(src_text) \
                   + ['<eos>']
        target_tokens = ['<sos>'] + self.target_tokenizer(target_text) \
                      + ['<eos>']

        src_ids = self.encode(src_tokens, self.src_vocab)
        target_ids = self.encode(target_tokens, self.target_vocab)

        return torch.tensor(src_ids), torch.tensor(target_ids)

# handles the complete data pipeline
class TranslationData:
    def __init__(self, src_lang='en', target_lang='fr',
                 max_vocab_size=20000, batch_size=32,
                 max_len=100, src_vocab=None, target_vocab=None):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.max_vocab_size = max_vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        
        try:
            self.spacy_src = spacy.load(f'{src_lang}_core_web_sm')
            self.spacy_target = spacy.load(f'{target_lang}_core_news_sm')
        except:
            raise RuntimeError('Spacy model failed to load')

        self.special_tokens = {'<pad>': 3, '<sos>': 1, '<eos>': 2,
                               '<unk>': 0}
        
        # pass the vocab if used during test post model development
        # so we can reuse the same vocab
        self.src_vocab = src_vocab
        self.target_vocab = target_vocab
        # variables to store the loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def tokenize_src(self, text):
        return [token.text.lower() for token in
                self.spacy_src.tokenizer(text)]
    
    def tokenize_target(self, text):
        return [token.text.lower() for token in
                self.spacy_target.tokenizer(text)]
    
    # Builds a vocabulary dictionary mapping token -> index
    def build_vocab_OLD(self, sentences, tokenizer):
        counter = Counter()
        for sentence in sentences:
            tokens = tokenizer(sentence)
            counter.update(tokens)
        vocab = dict(self.special_tokens)
        idx = len(vocab)
        for word, freq in counter.most_common(self.max_vocab_size - len(vocab)):
            vocab[word] = idx
            idx += 1

        return vocab
    
    def build_vocab(self, dataset_split, tokenizer, text_field='translation',
                    lang='en'):
        # create a counter for future frequency calculations
        counter = Counter()
        
        # function to tokenize in parallel using .map
        def tokenize_fn(batch):
            return {'tokens': [tokenizer(sample[lang]) for sample in batch[text_field]]}
        
        # apply map with batched=True for faster processing
        tokenized = dataset_split.map(
            tokenize_fn, batched = True, batch_size = 1000, num_proc=4,
            remove_columns=dataset_split.column_names # keep only tokens
        )

        # Flatten all tokens
        for batch in tokenized:
            counter.update(batch['tokens'])

        # build vocab based on frequency
        vocab = dict(self.special_tokens) # start with the special tokens
        idx = len(vocab) # start after the special tokens end
        for word, freq in counter.most_common(self.max_vocab_size - len(vocab)):
            vocab[word] = idx
            idx += 1

        return vocab
    
    # Pads batch of source and target sentences to uniform length.
    # batch: contains a list of (src, target) pairs
    def collate_fn(self, batch):
        src_batch, target_batch = zip(*batch)
        src_batch = pad_sequence(src_batch,
                                 padding_value=self.special_tokens['<pad>'],
                                 batch_first=True)
        target_batch = pad_sequence(target_batch,
                                    padding_value=self.special_tokens['<pad>'],
                                    batch_first=True)
        return src_batch, target_batch
    
    def prepare_data(self):
        print('Loading dataset...')
        dataset = load_dataset('iwslt2017', 'iwslt2017-en-fr', trust_remote_code=True)
        train_data = dataset['train']
        valid_data = dataset['validation']
        test_data = dataset['test']

        if self.src_vocab is None or self.target_vocab is None:
            print("Building vocabularies from training data...")
            self.src_vocab = self.build_vocab(train_data, self.tokenize_src,
                                            text_field='translation',
                                            lang=self.src_lang)
            self.target_vocab = self.build_vocab(train_data, self.tokenize_target,
                                             text_field='translation',
                                             lang=self.target_lang)

            print(f'Vocab sizes: src = {len(self.src_vocab)},' +
                f'target = {len(self.target_vocab)}')
        else:
            print('Using provided vocabularies')
        
        # create datasets
        train_dataset = TranslationDataset(data=train_data, src_lang=self.src_lang, target_lang=self.target_lang,
                                           src_tokenizer=self.tokenize_src, target_tokenizer=self.tokenize_target,
                                           src_vocab=self.src_vocab, target_vocab=self.target_vocab,
                                           max_len=self.max_len)
        valid_dataset = TranslationDataset(data=valid_data, src_lang=self.src_lang, target_lang=self.target_lang,
                                           src_tokenizer=self.tokenize_src, target_tokenizer=self.tokenize_target,
                                           src_vocab=self.src_vocab, target_vocab=self.target_vocab,
                                           max_len=self.max_len)
        test_dataset = TranslationDataset(data=test_data, src_lang=self.src_lang, target_lang=self.target_lang,
                                          src_tokenizer=self.tokenize_src, target_tokenizer=self.tokenize_target,
                                           src_vocab=self.src_vocab, target_vocab=self.target_vocab,
                                           max_len=self.max_len)
        # create dataloaders
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.collate_fn)
        self.valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size,
                                      shuffle=False, collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                      shuffle=False, collate_fn=self.collate_fn)
        
        print('Data Loaders ready')

    def get_dataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader
    
    def get_vocabs(self):
        return self.src_vocab, self.target_vocab