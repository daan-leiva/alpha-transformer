import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing: float, vocab_size: int, ignore_index=-100):
        super().__init__()
        self.ε = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # pred: (batch_size, seq_len, vocab_size)
        # target: (batch_size, seq_len)
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.ε / (self.vocab_size - 1))
            mask = target != self.ignore_index
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.ε)
            true_dist[~mask] = 0

        log_probs = F.log_softmax(pred, dim=-1)
        return -(true_dist * log_probs).sum(dim=1).mean()
