import numpy as np
from numpy.typing import NDArray
from typing import Literal, Callable, Dict, List, Tuple

from src.sem.abstract import StructuralEquationModel as SEM


class NonlinearSimulationSEM(SEM):
    _FUNCTIONS: Dict[str, Callable[[float], float]] = {
        'abs': lambda x: np.abs(x),
        'sin': lambda x: np.sin(x),
        'step': lambda x: np.sign(x),
    }
    
    def __init__(
            self, function_name: Literal['abs', 'sin', 'step']='abs'
        ):
        self.function_name = function_name
        self._f = self.get_function(function_name)
        super(NonlinearSimulationSEM, self).__init__()

    def sample(self, N: int=1, intervention: bool=False, **kwargs) -> Tuple[NDArray, NDArray]:
        C = np.random.randn(N, 1)
        
        N_X = np.random.randn(N, 1)
        if intervention:
            X = np.random.randn(N, 1)
        else:
            X = 2.0 * C + N_X
        
        N_Y = np.random.randn(N, 1)
        Y = self.f(X) + 5.0 * C + (2.0 ** -0.5) * N_Y
        return X, Y
    
    def f(self, X) -> NDArray:
        return self._f(X)
    
    @property
    def solution(self):
        raise NotImplementedError()
    
    @classmethod
    def num_functions(cls) -> int:
        return len(cls._FUNCTIONS)
    
    @classmethod
    def get_function(cls, function_name) -> Callable[[float], float]:
        return cls._FUNCTIONS[function_name]
    
    @classmethod
    def get_functions(cls) -> List[Literal['abs', 'sin', 'step']]:
        return list(cls._FUNCTIONS.keys())
