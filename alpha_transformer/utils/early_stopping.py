class EarlyStopping:
    """
    Implements early stopping to halt training when a monitored metric stops improving.

    Args:
        patience (int): Number of evaluations to wait after last improvement before stopping.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        mode (str): 'min' if the metric should decrease (e.g., loss),
                    'max' if the metric should increase (e.g., accuracy or BLEU).
    """

    def __init__(self, patience=3, min_delta=1e-4, mode='min'):
        if mode not in ['min', 'max']:
            raise ValueError(f'Invalid mode "{mode}". Choose "min" or "max".')

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_metric = None
        self.patience_counter = 0
        self.early_stopped = False

    def step(self, metric):
        """
        Call this method after each evaluation to check if training should stop.

        Args:
            metric (float): Current value of the monitored metric.

        Returns:
            bool: True if early stopping condition is met, else False.
        """
        if self.best_metric is None:
            self.best_metric = metric
            return False

        if self._is_improved(metric):
            self.best_metric = metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stopped = True
                return True
            return False

    def _is_improved(self, metric):
        """
        Checks whether the new metric is an improvement based on the mode.

        Returns:
            bool: True if improved.
        """
        if self.mode == 'max':
            return metric > self.best_metric + self.min_delta
        else:  # mode == 'min'
            return metric < self.best_metric - self.min_delta

    def reset(self):
        """
        Resets the internal state for reuse.
        """
        self.best_metric = None
        self.patience_counter = 0
        self.early_stopped = False
