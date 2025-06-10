import matplotlib.pyplot as plt

class TrainingPlotter:
    def __init__(self, training_losses, val_losses, val_bleu_scores):
        self.training_losses = training_losses
        self.val_losses = val_losses
        self.val_bleu_scores = val_bleu_scores
        self.fig = None

    def create_plot(self):
        epochs = range(1, len(self.training_losses) + 1)

        # Loss plots
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(epochs, self.training_losses, label='Train Loss', color='tab:blue')
        ax1.plot(epochs, self.val_losses, label='Val Loss', color='tab:orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')

        # BLEU plots on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.val_bleu_scores, label='BLEU Score', color='tab:green')
        ax2.set_ylabel('BLEU Score')
        ax2.legend(loc='upper right')

        plt.title("Training Loss and BLEU Score vs Epochs")
        
        self.fig = fig

    def plot(self):
        if self.fig is None:
            self.create_plot()
        self.fig.show()

    def save(self, filename='training_curves.png'):
        if self.fig is None:
            self.create_plot()
        self.fig.savefig(filename)
        plt.close(self.fig) # to avoid memory leaks