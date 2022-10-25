import numpy as np
import scipy as sc
from scipy.optimize import minimize
from sklearn.model_selection import RandomizedSearchCV

from src.regressors.abstract import ModelSelector
from src.regressors.erm import LeastSquaresClosedForm as ERM


class LeaveOneOut(ModelSelector, RandomizedSearchCV):
    def __init__(self, **kwargs):
        super(LeaveOneOut, self).__init__(**kwargs)


class ConfounderCorrection(ModelSelector):
    def __init__(self, estimator, **kwargs):
        self.best_estimator_ = estimator
    
    def fit(self, X, y, **kwargs):
        _, d = X.shape
        X_ = kwargs["GX"]
        Cxx = np.cov(X_, rowvar=False)
        W_erm = ERM().fit(X_, y).solution
        sqnorm =  (1 - self.estimate_beta(Cxx, W_erm)) * (W_erm**2).sum()

        self.best_estimator_.alpha = minimize(self._sqnorm_diff,
                                              [10.0],
                                              bounds = [(0, None)],
                                              args=(sqnorm, X, y, *kwargs.values())).x.item()

        self.best_estimator_.fit(X, y, **kwargs)
        return self

    def _sqnorm_diff(self, alpha, sqnorm, X, y, *args):
        self.best_estimator_.alpha = alpha.item()
        norm = np.sqrt((self.best_estimator_.fit(X, y, *args).solution**2).sum())
        return (norm - np.sqrt(sqnorm))**2

    @staticmethod
    def density(linear_map, vector):
        d = vector.shape[0]
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
            theta_est = minimize(cls.loglikelihood,
                                 0,
                                 bounds = [(0,None)],
                                 args = (Cxx,lsq_reg)).x
        except:
            theta_est = minimize(cls.loglikelihood,
                                 0,
                                 bounds = [(1e-9,None)],
                                 args = (Cxx,lsq_reg)).x
        return theta_est
    
    @classmethod
    def loglikelihood(cls, theta, cov, vector):
        d = vector.shape[0]
        matrix_squared = np.identity(d) + theta*np.linalg.inv(cov)
        matrix = sc.linalg.sqrtm(matrix_squared)
        return - np.log(cls.density(matrix,vector))

