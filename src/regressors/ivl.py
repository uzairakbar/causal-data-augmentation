import torch
import numpy as np
import cvxpy as cp
from math import comb
from loguru import logger
import torch.nn.functional as F

from src.regressors.utils import Model, device

from src.regressors.abstract import RegressorIVlike as IVL

from src.regressors.erm import (
    GradientDescentERM as ERM,
    LeastSquaresClosedForm as OLS,
)
from src.regressors.iv import GeneralizedMomentMethodIV as IV


DEVICE: str=device()
MAX_BATCH: int=256
LOG_FREQUENCY: int=100


class LeastSquaresClosedFormIVlike(IVL):
    def __init__(self, alpha = 1.0):
        super(LeastSquaresClosedFormIVlike, self).__init__(alpha)
        
    def _fit(self, X, y, G=None, GX=None, **kwargs):
        N = len(GX)
        I = np.eye(N)
        Cgg = G.T @ G
        alpha = self._alpha
        PI_G = G @ np.linalg.pinv( Cgg ) @ G.T
        
        # IV + a * ERM
        K = (
            (np.sqrt( 1+alpha ) - np.sqrt(alpha)) * PI_G
            +
            np.sqrt(alpha) * I
        )
        X_, y_ = K @ GX, K @ y
        
        self._W = OLS().fit(X_, y_).solution

        return self
    
    def _predict(self, X):
        return X @ self._W


class ProjectedLeastSquaresIVlike(IVL):
    def __init__(self, alpha = None):
        super(ProjectedLeastSquaresIVlike, self).__init__(alpha)

    def _fit(self, X, y, G=None, GX=None, **kwargs):
        h_erm = OLS().fit(GX, y).solution

        s1 = OLS().fit(G, GX).solution
        GXhat = G @ s1
        
        A = GXhat.T @ GXhat
        b = GXhat.T @ y

        h = cp.Variable(h_erm.shape)
        cost = cp.norm(h - h_erm)
        constraints = [A@h == b]
        prob = cp.Problem(
            cp.Minimize(cost),
            constraints
        )
        try:
            result = prob.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            result = prob.solve(solver=cp.ECOS)
        self._W = h.value

        return self
    
    def _predict(self, X):
        return X @ self._W


class ConstrainedLeastSquaresIVlike(IVL):
    def __init__(self, alpha = None):
        super(ConstrainedLeastSquaresIVlike, self).__init__(alpha)

    def _fit(self, X, y, G=None, GX=None, **kwargs):
        h_erm = OLS().fit(GX, y).solution

        s1 = OLS().fit(G, GX).solution
        GXhat = G @ s1
        
        A = GXhat.T @ GXhat
        b = GXhat.T @ y

        h = cp.Variable(h_erm.shape)
        cost = cp.norm(GX@h - y)
        constraints = [A @ h == b]
        prob = cp.Problem(
            cp.Minimize(cost),
            constraints
        )
        try:
            result = prob.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            result = prob.solve(solver=cp.ECOS)
        self._W = h.value
        return self
    
    def _predict(self, X):
        return X @ self._W


class GeneralizedMomentMethodIVlike(IV, ERM):
    def __init__(self,
                 model: Model='linear',
                 alpha=0.1):
        self.alpha = alpha
        self.model = model  # TODO: refactor code
        super(GeneralizedMomentMethodIVlike,
                     self).__init__(model=model)
    
    def fit(self, X, y, G, GX, **kwargs):
        return super(GeneralizedMomentMethodIVlike,
                        self).fit(X=GX, y=y, Z=G, **kwargs)
    
    def loss(self,
             X, y, G):
        N, M = X.shape
        I = torch.eye(N).to(DEVICE)
        Pi = G @ torch.linalg.pinv( G.t() @ G ) @ G.t()

        mse = F.mse_loss(self.f(X), y, reduction='none')

        # IV loss := Pi @ mse
        # ERM loss := mse
        #   => IVL loss := IV + a * ERM
        ivl_a_loss = ((Pi + self.alpha * I) @ mse).mean()
        self._optimizer.zero_grad()
        ivl_a_loss.backward()
        return ivl_a_loss
