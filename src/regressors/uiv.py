import torch
import numpy as np
import cvxpy as cp
from math import comb
from loguru import logger
import torch.nn.functional as F

from src.regressors.utils import Model, device

from src.regressors.abstract import RegressorUnfaithfulIV as UIV

from src.regressors.erm import GradientDescentERM as ERM
from src.regressors.erm import LeastSquaresClosedForm as OLS
from src.regressors.iv import GeneralizedMomentMethodIV as IV


DEVICE: str=device()
MAX_BATCH: int=2_500
LOG_FREQUENCY: int=100


class LeastSquaresClosedFormUnfaithfulIV(UIV):
    def __init__(self, alpha = 1.0):
        super(LeastSquaresClosedFormUnfaithfulIV, self).__init__(alpha)
        
    def _fit(self, X, y, G=None, GX=None, **kwargs):
        N = len(GX)
        I = np.eye(N)
        Cgg = G.T @ G
        PI_G = G @ np.linalg.pinv( Cgg ) @ G.T
        
        # IV + a * ERM
        K = (PI_G + np.sqrt(self._alpha) * I)
        X_, y_ = K @ GX, K @ y
        
        self._W = OLS().fit(X_, y_).solution

        return self
    
    def _predict(self, X):
        return X @ self._W


class ProjectedLeastSquaresUnfaithfulIV(UIV):
    def __init__(self, alpha = None):
        super(ProjectedLeastSquaresUnfaithfulIV, self).__init__(alpha)

    def _fit(self, X, y, G=None, GX=None):
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


class ConstrainedLeastSquaresUnfaithfulIV(UIV):
    def __init__(self, alpha = None):
        super(ConstrainedLeastSquaresUnfaithfulIV, self).__init__(alpha)

    def _fit(self, X, y, G=None, GX=None):
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


class GeneralizedMomentMethodUnfaithfulIV(IV, ERM):
    def __init__(self,
                 model: Model='linear',
                 alpha=0.1):
        self.alpha = alpha
        self.model = model  # TODO: refactor code
        super(GeneralizedMomentMethodUnfaithfulIV,
                     self).__init__(model=model)
    
    def fit(self, X, y, G, GX, **kwargs):
        return super(GeneralizedMomentMethodUnfaithfulIV,
                        self).fit(X=GX, y=y, Z=G, **kwargs)
    
    def loss(self,
             X, y, G):
        N, M = X.shape
        I = torch.eye(N).to(DEVICE)
        Pi = G @ torch.linalg.pinv( G.t() @ G ) @ G.t()

        mse = F.mse_loss(self.f(X), y, reduction='none')

        # IV loss := Pi @ mse
        # ERM loss := mse
        #   => UIV loss := IV + a * ERM
        uiv_a_loss = ((Pi + self.alpha * I) @ mse).mean()
        self._optimizer.zero_grad()
        uiv_a_loss.backward()
        return uiv_a_loss
