import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Instead of training on a hard one hot target, this loss spreads a small
    amount of probability mass over all incorrect classes. This can make
    training less brittle and reduce over confidence in the model.

    For a target class c, the smoothed distribution is

        p_c       = 1 - epsilon
        p_else    = epsilon / (vocab_size - 1)

    except for positions that are ignored, such as padding tokens.
    """

    def __init__(self, label_smoothing: float, vocab_size: int, ignore_index=-100):
        """
        Parameters
        ----------
        label_smoothing : float
            Smoothing factor epsilon, usually in the range [0.0, 0.2].
        vocab_size : int
            Number of classes, usually the vocabulary size.
        ignore_index : int
            Target index to ignore when computing the loss, such as the pad token.
        """
        super().__init__()
        self.ε = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Compute the label smoothed negative log likelihood loss.

        Parameters
        ----------
        pred : Tensor
            Raw predictions (logits) with shape (batch_size, seq_len, vocab_size)
            or flattened to (batch_size * seq_len, vocab_size).
        target : Tensor
            Ground truth indices with shape (batch_size, seq_len) or flattened
            to (batch_size * seq_len,).

        Returns
        -------
        Tensor
            Scalar loss value averaged over non ignored positions.
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
