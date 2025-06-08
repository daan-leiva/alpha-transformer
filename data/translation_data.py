import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# provides data for training
class TranslationDataset(Dataset):
    def __init__(self, data, src_lang, tgt_lang, src_tokenizer,
                    tgt_tokenizer, special_tokens,
                    max_len=100):
        super().__init__()
        # dictionary of data
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.special_tokens = special_tokens
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
        tgt_text = sample[self.tgt_lang]

        # get special tokens
        sos_id = self.special_tokens['<sos>']
        eos_id = self.special_tokens['<eos>']

        src_ids = [sos_id] + self.src_tokenizer.encode(src_text, out_type=int) + [eos_id]
        tgt_ids = [sos_id] + self.tgt_tokenizer.encode(tgt_text, out_type=int) + [eos_id]

        return torch.tensor(src_ids[:self.max_len]), torch.tensor(tgt_ids[:self.max_len])

# handles the complete data pipeline
class TranslationData:
    def __init__(self, src_lang='en', tgt_lang='fr',
                 batch_size=32, max_len=100,
                 src_tokenizer=None, tgt_tokenizer=None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.max_len = max_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.special_tokens = {'<pad>': 3, '<sos>': 1, '<eos>': 2,
                               '<unk>': 0}
        
        # variables to store the loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    # Pads batch of source and target sentences to uniform length.
    # batch: contains a list of (src, tgt) pairs
    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = pad_sequence(src_batch,
                                 padding_value=self.special_tokens['<pad>'],
                                 batch_first=True)
        tgt_batch = pad_sequence(tgt_batch,
                                    padding_value=self.special_tokens['<pad>'],
                                    batch_first=True)
        return src_batch, tgt_batch
    
    def prepare_data(self):
        print('Loading dataset...')
        dataset = load_dataset('iwslt2017', 'iwslt2017-en-fr', trust_remote_code=True)
        train_data = dataset['train']
        valid_data = dataset['validation']
        test_data = dataset['test']
        
        # create datasets
        train_dataset = TranslationDataset(data=train_data, src_lang=self.src_lang, tgt_lang=self.tgt_lang,
                                           src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer,
                                           max_len=self.max_len)
        valid_dataset = TranslationDataset(data=valid_data, src_lang=self.src_lang, tgt_lang=self.tgt_lang,
                                           src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer,
                                           max_len=self.max_len)
        test_dataset = TranslationDataset(data=test_data, src_lang=self.src_lang, tgt_lang=self.tgt_lang,
                                           src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer,
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