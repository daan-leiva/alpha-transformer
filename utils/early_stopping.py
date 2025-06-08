class EarlyStopping():
    def __init__(self, patience=3, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.patience_counter = 0
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = None
        self.early_stopped = False
        if mode not in ['min', 'max']:
            raise ValueError(f'Invalid mode {mode} passed to EarlyStopping. Use "min" or "max" only.')
        
    def step(self, metric):
        # check if first call
        if self.best_metric is None:
            self.best_metric = metric
            return False
        
        # check for improvement
        if self.is_improved(metric):
            self.patience_counter = 0
            self.best_metric = metric
            return False
        else:   
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stopped = True
                return True
            return False
                
    def is_improved(self, metric):
        if self.mode == 'max':
            return metric > self.best_metric + self.min_delta
        else: # mode = 'min'
            return metric < self.best_metric - self.min_delta