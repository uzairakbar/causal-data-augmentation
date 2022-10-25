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


class ModelSelector(ABC, BaseSearchCV):
    def __init__(self, **kwargs):
        r2 = make_scorer(self.r2)
        super(ModelSelector, self).__init__(**kwargs, scoring=r2)
    
    @property
    def alpha(self):
        return self.best_estimator_.alpha
    
    @property
    def solution(self):
        return self.best_estimator_.solution

    @staticmethod
    def r2(y, yhat, **kwargs):
        ss = lambda x: sum(x**2)
        ss_residual = ss(y - yhat)
        ss_total = ss(y - y.mean())
        r2 = 1 - ss_residual/ss_total
        return r2

