import matplotlib.pyplot as plt

class TrainingPlotter:
    """
    Plots training loss, validation loss, and BLEU scores over epochs.

    Args:
        training_losses (list of float): Training loss values.
        val_losses (list of float): Validation loss values.
        val_bleu_scores (list of float): BLEU scores for validation data.
    """

    def __init__(self, training_losses, val_losses, val_bleu_scores):
        assert len(training_losses) == len(val_losses) == len(val_bleu_scores), \
            "All metric lists must be of the same length"
        
        self.training_losses = training_losses
        self.val_losses = val_losses
        self.val_bleu_scores = val_bleu_scores
        self.fig = None

    def create_plot(self):
        epochs = range(1, len(self.training_losses) + 1)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot training and validation loss
        ax1.plot(epochs, self.training_losses, label='Train Loss', color='tab:blue', marker='o')
        ax1.plot(epochs, self.val_losses, label='Val Loss', color='tab:orange', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot BLEU score on a second y-axis
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.val_bleu_scores, label='BLEU Score', color='tab:green', marker='^')
        ax2.set_ylabel('BLEU Score')
        ax2.legend(loc='upper right')

        plt.title("Training Loss and BLEU Score over Epochs")
        self.fig = fig

    def plot(self):
        """Displays the plot (creates it if not already generated)."""
        if self.fig is None:
            self.create_plot()
        self.fig.tight_layout()
        self.fig.show()

    def save(self, filename='training_curves.png'):
        """Saves the plot to a file."""
        if self.fig is None:
            self.create_plot()
        self.fig.tight_layout()
        self.fig.savefig(filename, dpi=300)
        plt.close(self.fig)  # Free up memory
