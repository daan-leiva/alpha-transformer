import torch
import torch.nn as nn
from torch import optim
from transformer.transformer import Transformer
from data.translation_data import TranslationData
import argparse
import os
import datetime
from trainer import Trainer
import sentencepiece as spm

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Training Script')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--save_path', type=str, default=now_str)
    parser.add_argument('--max_len', type=int, default=100)
    args = parser.parse_args()
    return args

# get the model object
def create_model(vocab_size, d_model=64, n_heads=4, max_len=50, dropout_rate=0.1, encoding_type='sinusoidal', hidden_ff_d=128,
                 num_layers=2):
    return Transformer(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                       max_len=max_len, dropout_rate = dropout_rate,
                       encoding_type=encoding_type, hidden_ff_d=hidden_ff_d,
                       num_encoder_layers=num_layers, num_decoder_layers=num_layers)

# main loop
def main():
    # get arguments
    args = parse_args()
    # model path
    model_path = f'./checkpoints/{args.save_path}'
    # create save data file if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    # create a log file path
    log_file_path = f'{model_path}/train_log.txt'
    log_file = open(log_file_path, 'w')
    # hyperparameters
    src_lang = 'en'
    tgt_lang = 'fr'
    d_model = args.d_model
    n_heads = args.n_heads
    num_layers = args.num_encoder_layers
    hidden_ff_d = d_model * 4
    max_len = args.max_len
    dropout_rate = args.dropout_rate
    # create sentence piece
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load('data/spm.model')
    vocab_size = sp_tokenizer.get_piece_size()
    # test for gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create model
    model = create_model(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                         num_layers=num_layers, hidden_ff_d=hidden_ff_d,
                         max_len=max_len, dropout_rate=dropout_rate).to(device=device)
    # training parameters
    lr = args.learning_rate
    batch_size = args.batch_size
    
    # prepare data
    data_module = TranslationData(src_lang=src_lang, tgt_lang=tgt_lang,
                                  batch_size=batch_size, max_len=max_len,
                                  tokenizer=sp_tokenizer)
    data_module.prepare_data()
    # create training aids
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=data_module.special_tokens['<pad>'])
    # get loaders
    train_loader, valid_loader, _ = data_module.get_dataloaders()
    # creat a trainer object
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler,
                      loss_fn=loss_fn, train_loader=train_loader, val_loader=valid_loader,
                      args=args, log_file=log_file,
                      special_tokens=data_module.special_tokens, tokenizer=sp_tokenizer)
    trainer.train()
    # save the final model
    trainer.save_checkpoint(path=f'{model_path}/final_model.pt')
    # close log file
    log_file.close()
    
if __name__ == '__main__':
    main() 