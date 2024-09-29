import torch
import numpy as np
from torch import nn
from numpy.typing import NDArray
from typing import Optional, Literal, Callable, Dict
from sklearn.model_selection import BaseCrossValidator


Model = Literal['linear', '2-layer', 'cmnist', 'rmnist']


MODELS: Dict[Model, Callable[[int], nn.Sequential]] = {
    'linear': lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 1, bias=False)
    ),
    '2-layer': lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 20),
        nn.LeakyReLU(0.2),
        nn.Linear(20, 1)
    ),
    'cmnist': lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(True),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Linear(256, 1),
        nn.Sigmoid()
    ),
    'rmnist': lambda input_dim: nn.Sequential(
        nn.Unflatten(1, torch.Size([1, 28, 28])),
        nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False),
        nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(1, -1),
        nn.Linear(64 * 4 * 4, 128),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1)
    ),
}


class VanillaSplitter(BaseCrossValidator):
    def __init__(self, frac: float=0.2, **kwargs):
        self.frac = frac
        super(VanillaSplitter, self).__init__(**kwargs)
    
    def _iter_test_indices(
            self,
            X: NDArray,
            y: NDArray=None,
            groups: Optional[NDArray]=None
        ):
        n_samples = len(X)
        indices = np.arange(n_samples)
        n_test_samples = round(n_samples*self.frac)
        test_indices = np.random.choice(
            indices, size=n_test_samples, replace=False
        )
        yield test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


class LevelSplitter(BaseCrossValidator):
    def __init__(self, frac: float=0.2, **kwargs):
        self.frac = frac
        super(LevelSplitter, self).__init__(**kwargs)
    
    def _iter_test_indices(
            self,
            X: NDArray,
            y: NDArray=None,
            groups: Optional[NDArray]=None
        ):
        levels, indices, level_count = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        
        n_levels = len(levels)

        n_test_levels = round(n_levels*self.frac)

        p = level_count/sum(level_count)
        p_inverse = (1.0 - p)/(len(p) - 1.0)
        
        test_levels = np.random.choice(
            levels, size=n_test_levels, replace=False, p=p_inverse
        )

        test_indices = np.where(np.in1d(groups, test_levels))[0]

        yield test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device)
