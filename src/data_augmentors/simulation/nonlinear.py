import numpy as np
from numpy.typing import NDArray
from typing import Literal, Dict, Callable, Tuple

from src.data_augmentors.abstract import DataAugmenter as DA


FunctionType = Literal['abs', 'sin', 'step']
AugmentorType = Callable[[float, float], float]
SamplerType = Callable[[int], NDArray]


class NonlinearSimulationDA(DA):
    def __init__(self, function_name: FunctionType):
        AUGMENTATION: Dict[FunctionType, AugmentorType] = {
            'abs': lambda x, g: (g * x),
            'sin': lambda x, g: (x + 2*np.pi*g),
            'step': lambda x, g: (x * np.float_power(10, g)),
        }
        SAMPLER: Dict[FunctionType, SamplerType] = {
            'abs': lambda N: np.random.choice([-1, 1], size=(N, 1)),
            'sin': lambda N: np.random.choice([-1, 0, 1], size=(N, 1)),
            'step': lambda N: np.random.choice([-1, 0, 1], size=(N, 1)),
        }
        self.function_name = function_name
        self._augment = AUGMENTATION[function_name]
        self._sampler = SAMPLER[function_name]
    
    @property
    def augmentation(self):
        return 'nonlinear_simulation'
    
    def augment(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        N = len(X)
        G = self._sampler(N)

        GX = self._augment(X, G)
        
        return GX, G
