from alpha_transformer.transformer.transformer import Transformer
from alpha_transformer.utils import bleu_scorer

def main():
    """
    Simple sanity check script.

    Confirms that:
      transformer.transformer.Transformer can be imported and instantiated
      utils.bleu_scorer module is importable
    """
    # Confirm imports succeeded
    print("Successfully imported Transformer and BLEU scorer.")

    # Instantiate Transformer model with dummy values
    model = Transformer(
        vocab=1000,             # Vocabulary size (dummy value)
        d_model=256,            # Embedding dimension
        n_heads=4,              # Number of attention heads
        max_len=100,            # Max sequence length
        dropout_rate=0.1,       # Dropout rate
        hidden_ff_d=1024,       # Hidden dimension in feed-forward layers
        num_encoder_layers=6,   # Number of encoder layers
        num_decoder_layers=6    # Number of decoder layers
    )

    # Print confirmation of model instantiation
    print("Model instantiated:", model.__class__.__name__)

    # Confirm BLEU scorer interface is available
    has_bleu = hasattr(bleu_scorer, "calculate_bleu")
    print("BLEU scorer module available:", has_bleu)


if __name__ == "__main__":
    main()
