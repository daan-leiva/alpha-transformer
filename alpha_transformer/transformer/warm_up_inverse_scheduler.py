import torch

class WarmupInverseSquareRootScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and inverse square root decay.
    Used in Transformer training.

    Learning rate schedule:
        lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        d_model (int): Model dimensionality (used for scaling).
        last_epoch (int): The index of the last epoch. Default: -1.
    """
    def __init__(self, optimizer, warmup_steps, d_model, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self._step_count, 1)  # avoid division by zero
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
