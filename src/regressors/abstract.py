import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection._search import BaseSearchCV


class Regressor(ABC, BaseEstimator):
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @property
    def solution(self):
        return self._W

    def predict(self, X):
        return X @ self._W


class EmpiricalRiskMinimizer(Regressor):
    pass


class IVRegressor(Regressor):
    @abstractmethod
    def fit(self, X, y, Z):
        pass


class DAIVRegressor(IVRegressor, EmpiricalRiskMinimizer):
    def __init__(self, alpha=1.0):
        self._alpha = alpha
        super(DAIVRegressor, self).__init__()
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
    
    @abstractmethod
    def fit(self, X, y, G, GX):
        pass

