import numpy as np
from sklearn.model_selection import BaseCrossValidator


class VanillaSplitter(BaseCrossValidator):
    def __init__(self, frac=0.2, **kwargs):
        self.frac = frac
        super(VanillaSplitter, self).__init__(**kwargs)
    
    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        n_test_samples = round(n_samples*self.frac)
        test_indices = np.random.choice(
            indices, size=n_test_samples, replace=False
        )
        yield test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1
        
