import torch
import torch.nn as nn
from torch import optim
from transformer.transformer import Transformer
from data.translation_data import TranslationData
from transformer.label_smoothing_loss import LabelSmoothingLoss
from transformer.warm_up_inverse_scheduler import WarmupInverseSquareRootScheduler
import argparse
import os
import datetime
from trainer import Trainer
import sentencepiece as spm

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Training Script')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--save_path', type=str, default=now_str)
    parser.add_argument('--encoding_type', type=str, default='learnable')
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--src_lang', type=str, required=True, help='Source language (en)')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language (de, fr)')
    parser.add_argument('--sp_model_path', type=str, required=True, help='Path to SentencePiece model file (.model/.vocab)')
    parser.add_argument('--small_subset', action='store_true', help='Use small subset for faster debugging')
    parser.add_argument('--label_smoothing', action='store_true', help='If true use label smoothing instead')
    parser.add_argument('--warmup_scheduler', action='store_true', help='if true it uses the warm up inverse scheduler')
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
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    d_model = args.d_model
    n_heads = args.n_heads
    num_layers = args.num_layers
    hidden_ff_d = d_model * 4
    max_len = args.max_len
    dropout_rate = args.dropout_rate
    encoding_type = args.encoding_type
    # verify that it's value is sinusoidal or learnable
    if encoding_type not in ['sinusoidal', 'learnable']:
        raise ValueError("Encoding type can only be sinusoidal or learnable")

    # create sentence piece
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load(args.sp_model_path)
    vocab_size = sp_tokenizer.get_piece_size()
    # verify that the target language matches the file
    if args.tgt_lang not in args.sp_model_path:
        raise ValueError("The target language must match the sp model")

    # test for gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create model
    model = create_model(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
                         num_layers=num_layers, hidden_ff_d=hidden_ff_d,
                         max_len=max_len, dropout_rate=dropout_rate, encoding_type=encoding_type).to(device=device)
    # training parameters
    lr = args.learning_rate
    batch_size = args.batch_size
    
    # prepare data
    data_module = TranslationData(src_lang=src_lang, tgt_lang=tgt_lang,
                                  batch_size=batch_size, max_len=max_len,
                                  tokenizer=sp_tokenizer, small_subset=args.small_subset)
    data_module.prepare_data()
    # create training aids
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    if args.warmup_scheduler:
        scheduler = WarmupInverseSquareRootScheduler(optimizer, warmup_steps=4000, d_model=args.d_model)
        print("Warm up scheduler activate!")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if args.label_smoothing:
        loss_fn = LabelSmoothingLoss(label_smoothing=0.1, vocab_size=vocab_size, ignore_index=data_module.special_tokens['<pad>'])
        print("Smoothign Loss activated!")
    else:
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