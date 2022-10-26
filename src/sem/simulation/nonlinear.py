import numpy as np

from src.sem.abstract import StructuredEquationModel as SEM


class NonlinearSimulationSEM(SEM):
    _FUNCTIONS = {
        "abs": lambda x: np.abs(x),
        "sin": lambda x: np.sin(x),
        "step": lambda x: np.sign(x),
    }
    
    def __init__(self, function_name="abs"):
        self.function_name = function_name
        self._f = self.get_function(function_name)
        super(NonlinearSimulationSEM, self).__init__()

    def sample(self, N = 1, intervention = False, **kwargs):
        C = np.random.randn(N, 1)
        
        N_X = np.random.randn(N, 1)
        if intervention:
            X = np.random.randn(N, 1)
        else:
            X = 2.0 * C + N_X
        
        N_Y = np.random.randn(N, 1)
        Y = self.f(X) + 5.0 * C + (2.0 ** -0.5) * N_Y
        return X, Y
    
    def f(self, X):
        return self._f(X)
    
    @property
    def solution(self):
        raise NotImplementedError()
    
    @classmethod
    def num_functions(cls):
        return len(cls._FUNCTIONS)
    
    @classmethod
    def get_function(cls, function_name):
        return cls._FUNCTIONS[function_name]
    
    @classmethod
    def get_functions(cls):
        return list(cls._FUNCTIONS.keys())

