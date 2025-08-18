import warnings
import numpy as np
import scipy as sp
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    ParameterSampler,
    LeaveOneGroupOut,
    RandomizedSearchCV,
)

from src.regressors.abstract import ModelSelector
from src.regressors.erm import LeastSquaresClosedForm as ERM
from src.regressors.utils import (
    VanillaSplitter, LevelSplitter, RiskDifferenceScorer
)


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
        
        GX = kwargs.get('GX', X)
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


class RiskDifference(ModelSelector, RandomizedSearchCV):
    def __init__(
            self, 
            estimator, 
            param_distributions, 
            n_iter=10, 
            cv=5, 
            n_jobs=None, 
            **kwargs
        ):
        # make sure n_iter logic is correct
        if isinstance(
            param_distributions[next(iter(param_distributions))], 
            (list, np.ndarray)
        ):
            n_iter = len(
                param_distributions[next(iter(param_distributions))]
            )

        super(RiskDifference, self).__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=RiskDifferenceScorer(),
            cv=cv,
            n_jobs=n_jobs,
            **kwargs
        )

    def fit(self, X, y, X_A, y_A, **kwargs):
        groups = np.array(
            [0] * len(X) + [1] * len(X_A)
        )
        X_combined = np.vstack([X, X_A])
        y_combined = np.vstack(
            [y.reshape(-1, 1), y_A.reshape(-1, 1)]
        ).flatten()
        
        self.best_estimator_ = None
        self.best_score_ = -np.inf
        self.best_params_ = None
        self.cv_results_ = {'params': [], 'mean_test_score': []}

        splitter = KFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        param_sampler = list(
            ParameterSampler(
                self.param_distributions,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
        )

        # manual search over params because sklearn does not support 
        # custom params for CV scoring callable/function
        for params in param_sampler:
            scores = []
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='The groups parameter is ignored by KFold',
                    category=UserWarning
                )
                for train_idx, test_idx in splitter.split(X_combined, y_combined, groups):
                    estimator = clone(self.estimator).set_params(**params)
                    
                    X_train, y_train = X_combined[train_idx], y_combined[train_idx]
                    groups_train = groups[train_idx]
                    
                    X_test, y_test = X_combined[test_idx], y_combined[test_idx]
                    groups_test = groups[test_idx]

                    # fit the estimator
                    estimator._fit(X_train, y_train, groups=groups_train)

                    # score the estimator
                    score = self.scoring(
                        estimator, X_test, y_test, groups_test=groups_test
                    )
                    scores.append(score)
            
            mean_score = np.mean(scores)
            self.cv_results_['params'].append(params)
            self.cv_results_['mean_test_score'].append(mean_score)

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params

        # refit the best estimator
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
        self.best_estimator_._fit(X, y, X_A=X_A, y_A=y_A)
        
        return self
