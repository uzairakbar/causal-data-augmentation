import numpy as np

from src.data_augmentors.abstract import DataAugmenter as DA


class NonlinearSimulationDA(DA):
    def __init__(self, function_name):
        AUGMENTATION = {
            "abs": lambda x, g: (g * x),
            "sin": lambda x, g: (x + 2*np.pi*g),
            "step": lambda x, g: (x * 10**(g)),
        }
        SAMPLER = {
            "abs": lambda N: np.random.choice([-1, 1], size=(N, 1)),
            "sin": lambda N: np.random.choice([-1, 0, 1], size=(N, 1)),
            "step": lambda N: np.random.choice([-1, 0, 1], size=(N, 1)),
        }
        self.function_name = function_name
        self._augment = AUGMENTATION[function_name]
        self._sampler = SAMPLER[function_name]
    
    def augment(self, X):
        N = len(X)
        G = self._sampler(N)

        GX = self._augment(X, G)
        
        return GX, G

