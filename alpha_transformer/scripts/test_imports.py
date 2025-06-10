from transformer.transformer import Transformer
from utils import bleu_scorer

def main():
    print("Successfully imported Transformer and BLEU scorer.")

    # Optionally instantiate model with dummy values
    model = Transformer(
        vocab=1000,
        d_model=256,
        n_heads=4,
        max_len=100,
        dropout_rate=0.1,
        hidden_ff_d=1024,
        num_encoder_layers=6,
        num_decoder_layers=6
    )
    print("Model instantiated:", model.__class__.__name__)

    # Optionally test BLEU scorer interface
    print("BLEU scorer module available:", hasattr(bleu_scorer, "calculate_bleu"))

if __name__ == "__main__":
    main()
