import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection._search import BaseSearchCV


class Regressor(ABC, BaseEstimator):
    def fit(self, X, y, **kwargs):
        method_name = self.__class__.__name__
        if method_name == 'RICE':
            return self._fit(X, y, **kwargs)
        
        X = X.reshape(*X.shape[:1], -1)
        return self._fit(X, y, **kwargs)
    
    @abstractmethod
    def _fit(self, X, y):
        pass
    
    @property
    def solution(self):
        return self._W

    def predict(self, X):
        X = X.reshape(*X.shape[:1], -1)
        return self._predict(X)
    
    @abstractmethod
    def _predict(self, X):
        pass


class EmpiricalRiskMinimizer(Regressor):
    pass


class RegressorIV(Regressor):
    def fit(self, X, y, Z, **kwargs):
        return super(RegressorIV, self).fit(X=X, y=y, Z=Z, **kwargs)

    @abstractmethod
    def _fit(self, X, y, Z, **kwargs):
        pass


class RegressorUnfaithfulIV(Regressor):
    def __init__(self, alpha=1.0):
        self._alpha = alpha
        super(RegressorUnfaithfulIV, self).__init__()
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
    
    def fit(self, X, y, G, GX, **kwargs):
        # TODO: change this to simpler version for debugging!
        GX = GX.reshape(*GX.shape[:1], -1)
        X = X.reshape(*X.shape[:1], -1)
        # return self._fit(X, y, G, GX)
        return super(RegressorUnfaithfulIV, self).fit(X=X, y=y, G=G, GX=GX, **kwargs)
    
    @abstractmethod
    def _fit(self, X, y, G, GX, **kwargs):
        pass


class BaselineRegressor(Regressor):
    def __init__(self, alpha: float=1.0):
        self._alpha = alpha
        super(BaselineRegressor, self).__init__()
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
    
    def fit(self, X, y, Z=None, **kwargs):
        return super(BaselineRegressor, self).fit(X=X, y=y, Z=Z, **kwargs)

    @abstractmethod
    def _fit(self, X, y, Z, **kwargs):
        pass


class ModelSelector(ABC, BaseSearchCV):
    def __init__(self, metric='r2', **kwargs):
        if metric == 'r2':
            scoring = make_scorer(self.r2)
        elif metric == 'accuracy':
            scoring = make_scorer(self.accuracy)
        elif metric == 'mse':
            scoring = make_scorer(self.mse, greater_is_better=False)
        elif metric == 'cc':
            return super(ModelSelector, self).__init__(**kwargs)
        else:
            raise ValueError('Wrong value for validation metric.')
        return super(ModelSelector, self).__init__(**kwargs, scoring=scoring)
    
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
    
    @staticmethod
    def accuracy(y, yhat, **kwargs):
        return (y == yhat).mean()
    
    @staticmethod
    def mse(y, yhat, **kwargs):
        return (
            (y - yhat)**2
        ).mean()

