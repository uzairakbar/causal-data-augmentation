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

