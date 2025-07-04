import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Implements label smoothing to regularize the model by softening target distributions.

    Instead of using one-hot encoded targets, this loss assigns:
      - 1 - ε to the correct class
      - ε / (vocab_size - 1) to all other classes

    Args:
        label_smoothing (float): Smoothing factor ε (typically between 0.0 and 0.2)
        vocab_size (int): Number of classes (e.g., vocabulary size)
        ignore_index (int): Target index to ignore in loss (e.g., <pad> token)
    """
    def __init__(self, label_smoothing: float, vocab_size: int, ignore_index=-100):
        super().__init__()
        self.ε = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Computes the label-smoothed negative log-likelihood loss.

        Args:
            pred (Tensor): Raw predictions (logits) of shape (batch_size, seq_len, vocab_size)
            target (Tensor): Ground truth indices of shape (batch_size, seq_len)

        Returns:
            Tensor: Scalar loss value
        """
        # Flatten predictions and targets
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        # Create smoothed target distributions
        with torch.no_grad():
            true_dist = torch.full_like(pred, self.ε / (self.vocab_size - 1))  # Uniform distribution
            mask = target != self.ignore_index
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.ε)           # Assign 1 - ε to correct class
            true_dist[~mask] = 0                                               # Zero out ignored positions

        # Compute log-probabilities
        log_probs = F.log_softmax(pred, dim=-1)

        # Cross-entropy between log_probs and smoothed labels
        loss = -(true_dist * log_probs).sum(dim=1)

        # Return mean loss over non-ignored tokens
        return loss[mask].mean()
