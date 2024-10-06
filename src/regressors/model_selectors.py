import numpy as np
import scipy as sp
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
)

from src.regressors.abstract import ModelSelector
from src.regressors.erm import LeastSquaresClosedForm as ERM
from src.regressors.utils import VanillaSplitter, LevelSplitter


class VanillaCV(ModelSelector, RandomizedSearchCV):
    def __init__(self, frac=0.2, **kwargs):
        self.frac = frac
        super(VanillaCV, self).__init__(
            **kwargs, cv=VanillaSplitter(frac=frac)
        )


class LevelCV(ModelSelector, RandomizedSearchCV):
    def __init__(self, frac=0.2, **kwargs):
        self.frac = frac
        super(LevelCV, self).__init__(
            **kwargs, cv=LevelSplitter(frac=frac)
        )
    
    def fit(self, X, y, G=None, GX=None, Z=None, **kwargs):
        if G is None and Z is None:
            raise ValueError('Either Z or G has to be specified.')
        if G is not None:
            _, groups = np.unique(G, return_inverse=True, axis=0)
            return super(LevelCV, self).fit(X=X,
                                        y=y,
                                        G=G,
                                        GX=GX,
                                        groups=groups,
                                        **kwargs)
        else:
            _, groups = np.unique(Z, return_inverse=True, axis=0)
            return super(LevelCV, self).fit(X=X,
                                        y=y,
                                        Z=Z,
                                        GX=GX,
                                        groups=groups,
                                        **kwargs)


class LeaveOneOut(ModelSelector, RandomizedSearchCV):
    def __init__(self, **kwargs):
        super(LeaveOneOut, self).__init__(**kwargs)


class LeaveOneLevelOut(ModelSelector, RandomizedSearchCV):
    def __init__(self, **kwargs):
        super(LeaveOneLevelOut, self).__init__(**kwargs, cv=LeaveOneGroupOut())

    def fit(self, X, y, G=None, GX=None, Z=None, **kwargs):
        if G is None and Z is None:
            raise ValueError('Either Z or G has to be specified.')
        if G is not None:
            _, groups = np.unique(G, return_inverse=True, axis=0)
        else:
            _, groups = np.unique(Z, return_inverse=True, axis=0)

        return super(LeaveOneLevelOut, self).fit(X=X,
                                                 y=y,
                                                 G=G,
                                                 Z=Z,
                                                 GX=GX,
                                                 groups=groups,
                                                 **kwargs)


class ConfounderCorrection(ModelSelector, RandomizedSearchCV):
    def __init__(self, **kwargs):
        super(ConfounderCorrection, self).__init__(
            **kwargs, cv=VanillaSplitter(frac=0),
            metric='cc', scoring=self.cc_metric,
        )
    
    def fit(self, X, y, **kwargs):

        GX = kwargs['GX']
        Cxx = np.cov(GX, rowvar=False)
        W_erm = ERM().fit(GX, y).solution
        self.sqnorm =  (1 - self.estimate_beta(Cxx, W_erm)) * (W_erm**2).sum()

        return super(ConfounderCorrection, self).fit(X, y, **kwargs)
    

    def cc_metric(self, estimator, X, y_true, **kwargs):
        norm = np.sqrt((estimator.solution**2).sum())
        return -1.0 * (norm - np.sqrt(self.sqnorm))**2

    @staticmethod
    def density(linear_map, vector):
        d = len(vector)
        vector = vector/np.sqrt(sum(vector**2))
        vector_in = np.dot(np.linalg.inv(linear_map),vector)
        stretch_factor = np.sqrt(sum(vector_in**2))
        return 1/(np.linalg.det(linear_map)*(stretch_factor**d))
    
    @classmethod
    def estimate_beta(cls, Cxx, lsq_reg):
        d = len(lsq_reg)
        theta_est = cls.estimate_theta(Cxx,lsq_reg)
        Tinv = np.matrix.trace(np.linalg.inv(Cxx))/d
        beta_est = 1/(1+1/(0.001 + Tinv *theta_est))
        return beta_est.item()
    
    @classmethod
    def estimate_theta(cls, Cxx, lsq_reg):
        try:
            theta_est = sp.optimize.minimize(
                cls.loglikelihood,
                0,
                bounds = [(0,None)],
                args = (Cxx,lsq_reg)
            ).x
        except:
            theta_est = sp.optimize.minimize(
                cls.loglikelihood,
                0,
                bounds = [(1e-9,None)],
                args = (Cxx,lsq_reg)
            ).x
        return theta_est
    
    @classmethod
    def loglikelihood(cls, theta, cov, vector):
        d = len(vector)
        matrix_squared = np.identity(d) + theta*np.linalg.inv(cov)
        matrix = sp.linalg.sqrtm(matrix_squared)
        return - np.log(cls.density(matrix,vector))
