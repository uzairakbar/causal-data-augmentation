from abc import ABC, abstractmethod

class StructuredEquationModel(ABC):
    @abstractmethod
    def sample(self, N = 1, **kwargs):
        pass
    
    def __call__(self, N = 1, **kwargs):
        return self.sample(N = N, **kwargs)
    
    def f(self, X):
        return X @ self.W_XY
    
    @property
    def solution(self):
        return self.W_XY