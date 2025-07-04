import torch
import os
from trainer import Trainer
import argparse
from transformer.transformer import Transformer
from data.translation_data import TranslationData
import sentencepiece as spm

# ============================
# Argument Parser
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the .pt checkpoint to resume from')
    return parser.parse_args()

# ============================
# Load SentencePiece Tokenizer
# ============================
def load_tokenizer(sp_model_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(sp_model_path)
    return tokenizer

# ============================
# Main Resume Logic
# ============================
def main():
    args = parse_args()

    # Resolve project root directory dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Select appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint and recover training arguments
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    saved_args = checkpoint['args']
    save_dir = os.path.dirname(args.checkpoint_path)

    # Load SentencePiece tokenizer using saved path
    sp_model_path = os.path.join(root_dir, saved_args['sp_model_path'])
    tokenizer = load_tokenizer(sp_model_path=sp_model_path)

    # ============================
    # Rebuild Model from Args
    # ============================
    model = Transformer(
        vocab=tokenizer.get_piece_size(),
        d_model=saved_args['d_model'],
        n_heads=saved_args['n_heads'],
        max_len=saved_args['max_len'],
        dropout_rate=saved_args['dropout_rate'],
        hidden_ff_d=saved_args['d_model'] * 4,
        num_encoder_layers=saved_args['num_layers'],
        num_decoder_layers=saved_args['num_layers']
    ).to(device)

    # ============================
    # Rebuild Data Module
    # ============================
    data = TranslationData(
        src_lang=saved_args['src_lang'],
        tgt_lang=saved_args['tgt_lang'],
        batch_size=saved_args['batch_size'],
        max_len=saved_args['max_len'],
        tokenizer=tokenizer,
        small_subset=saved_args['small_subset']
    )
    data.prepare_data()
    train_loader, val_loader, _ = data.get_dataloaders()

    # ============================
    # Rebuild Optimizer and Scheduler
    # ============================
    optimizer = torch.optim.Adam(model.parameters(), lr=saved_args['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ============================
    # Loss Function
    # ============================
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data.special_tokens['<pad>'])

    # Open a resume training log file in append mode
    log_file = open(os.path.join(save_dir, "resume_training_log.txt"), "a")

    # ============================
    # Rebuild Trainer and Load State
    # ============================
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=saved_args,
        special_tokens=data.special_tokens,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        log_file=log_file
    )

    # Load checkpoint into Trainer and resume training
    trainer.load_checkpoint(args.checkpoint_path, need_optimizer=True, need_scheduler=True)
    trainer.train()

    # Close log file after training
    log_file.close()

# ============================
# Entry Point
# ============================
if __name__ == "__main__":
    main()
