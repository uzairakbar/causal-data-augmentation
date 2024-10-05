import torch
import numpy as np
import cvxpy as cp
from math import comb
import torch.nn.functional as F

from src.regressors.utils import Model, device

from src.regressors.abstract import DAIVRegressor
from src.regressors.erm import LeastSquaresClosedForm as OLS

from src.regressors.iv import IVGeneralizedMomentMethod as IV
from src.regressors.erm import LeastSquaresGradientDescent as ERM


DEVICE: str=device()
MAX_BATCH: int=2_500
LOG_FREQUENCY: int=100


class DAIVLeastSquaresClosedForm(DAIVRegressor):
    def __init__(self, alpha = 1.0):
        super(DAIVLeastSquaresClosedForm, self).__init__(alpha)
        
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


class DAIVProjectedLeastSquares(DAIVRegressor):
    def __init__(self, alpha = None):
        super(DAIVProjectedLeastSquares, self).__init__(alpha)

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
        result = prob.solve(solver=cp.CLARABEL)
        self._W = h.value

        return self
    
    def _predict(self, X):
        return X @ self._W


class DAIVConstrainedLeastSquares(DAIVRegressor):
    def __init__(self, alpha = None):
        super(DAIVConstrainedLeastSquares, self).__init__(alpha)

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
        result = prob.solve(solver=cp.CLARABEL)
        self._W = h.value
        return self
    
    def _predict(self, X):
        return X @ self._W


class DAIVGeneralizedMomentMethod(IV, ERM):
    def __init__(self,
                 model: Model='linear',
                 alpha=0.1):
        self.alpha = alpha
        self.model = model  # TODO: refactor code
        super(DAIVGeneralizedMomentMethod,
                     self).__init__(model=model)
    
    def fit(self, X, y, G, GX, **kwargs):
        return super(DAIVGeneralizedMomentMethod,
                     self).fit(X=GX, y=y, Z=G, **kwargs)
    
    def loss(self,
             X, y, G):
        N, M = X.shape
        I = torch.eye(N).to(DEVICE)
        Pi = G @ torch.linalg.pinv( G.t() @ G ) @ G.t()

        mse = F.mse_loss(self.f(X), y, reduction='none')

        # IV loss := Pi @ mse
        # ERM loss := mse
        #   => UIV loss := ERM + a * IV
        uiv_a_loss = ((I + self.alpha * Pi) @ mse).mean()
        self._optimizer.zero_grad()
        uiv_a_loss.backward()
        return uiv_a_loss
