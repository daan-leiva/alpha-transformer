import matplotlib.pyplot as plt

class TrainingPlotter:
    """
    Utility for plotting training loss, validation loss, and BLEU score over epochs.

    This is used at the end of training to generate a summary figure that is
    saved alongside checkpoints.
    """

    def __init__(self, training_losses, val_losses, val_bleu_scores):
        """
        Parameters
        ----------
        training_losses : list[float]
            Training loss per epoch.
        val_losses : list[float]
            Validation loss per epoch.
        val_bleu_scores : list[float]
            Validation BLEU score per epoch.
        """
        assert len(training_losses) == len(val_losses) == len(val_bleu_scores), \
            "All metric lists must be of the same length"
        
        self.training_losses = training_losses
        self.val_losses = val_losses
        self.val_bleu_scores = val_bleu_scores
        self.fig = None

    def create_plot(self):
        """
        Build the matplotlib figure but do not display or save it.

        Creates a figure with loss curves on the left axis and BLEU scores on
        a secondary right axis.
        """
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
        """
        Display the plot in an interactive session.

        If the figure has not been created yet, build it first.
        """
        if self.fig is None:
            self.create_plot()
        self.fig.tight_layout()
        self.fig.show()

    def save(self, filename: str = 'training_curves.png'):
        """
        Save the plot to disk.

        Parameters
        ----------
        filename : str
            File path where the figure should be saved.
        """
        if self.fig is None:
            self.create_plot()
        self.fig.tight_layout()
        self.fig.savefig(filename, dpi=300)
        # Close the figure to free resources when running many experiments
        plt.close(self.fig)
