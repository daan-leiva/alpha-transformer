import torch

class WarmupInverseSquareRootScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate schedule with a warmup phase followed by inverse square root decay.

    This follows the schedule from the original Transformer paper:

        lr(step) = d_model ** (-0.5)
                   * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    where step is the current optimizer step count.

    During the first warmup_steps updates the learning rate increases linearly.
    After warmup it decays in proportion to step ** (-0.5).
    """
    def __init__(self, optimizer, warmup_steps, d_model, last_epoch=-1):
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Wrapped optimizer whose learning rate will be scheduled.
        warmup_steps : int
            Number of warmup steps before switching to inverse square root decay.
        d_model : int
            Model dimensionality used for scaling the base learning rate.
        last_epoch : int
            Index of the last epoch, used when resuming training.
        """
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the current learning rate for each parameter group.

        Returns
        -------
        list[float]
            List of learning rates, one per parameter group.
        """
        # Use at least one step to avoid division by zero
        step = max(self._step_count, 1)  # avoid division by zero

        # Scale factor from the Transformer schedule
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        
        return [base_lr * scale for base_lr in self.base_lrs]
