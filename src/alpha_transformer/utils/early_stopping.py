class EarlyStopping:
    """
    Simple early stopping utility for training loops.

    It monitors a metric, such as validation loss or BLEU score, and signals
    when training should stop because the metric has not improved for a given
    number of evaluations.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Parameters
        ----------
        patience : int
            Number of evaluations to wait after the last improvement before
            requesting that training stop.
        min_delta : float
            Minimum change in the metric to qualify as an improvement.
        mode : str
            "min" if the metric is better when it is smaller, such as loss.
            "max" if the metric is better when it is larger, such as accuracy
            or BLEU.
        """
        if mode not in ['min', 'max']:
            raise ValueError(f'Invalid mode "{mode}". Choose "min" or "max".')

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_metric = None
        self.patience_counter = 0
        self.early_stopped = False

    def step(self, metric: float) -> bool:
        """
        Update the internal state with a new metric value.

        Parameters
        ----------
        metric : float
            Current value of the monitored metric.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if self.best_metric is None:
            # First observation initializes the baseline
            self.best_metric = metric
            return False

        if self._is_improved(metric):
            # Metric improved enough, reset the counter
            self.best_metric = metric
            self.patience_counter = 0
            return False
        else:
            # No improvement, increment the counter
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stopped = True
                return True
            return False

    def _is_improved(self, metric: float) -> bool:
        """
        Check whether the new metric is an improvement based on the mode.

        Returns
        -------
        bool
            True if the new metric is considered an improvement.
        """
        if self.mode == 'max':
            return metric > self.best_metric + self.min_delta
        else:  # mode == 'min'
            return metric < self.best_metric - self.min_delta

    def reset(self) -> None:
        """
        Reset internal state so the same instance can be reused.
        """
        self.best_metric = None
        self.patience_counter = 0
        self.early_stopped = False
