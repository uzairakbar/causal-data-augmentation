import numpy as np

from src.regressors.abstract import DAIVRegressor
from src.regressors.erm import LeastSquaresClosedForm as OLS


class DAIVLeastSquaresClosedForm(DAIVRegressor):
    def __init__(self, alpha = 1.0):
        super(DAIVLeastSquaresClosedForm, self).__init__(alpha)
        
    def fit(self, X, y, G=None, GX=None):
        N = len(GX)
        I = np.eye(N)
        Cgg = G.T @ G
        PI_G = G @ np.linalg.inv( Cgg ) @ G.T
        
        K = (I + np.sqrt(self._alpha) * PI_G)
        X_, y_ = K @ GX, K @ y
        
        self._W = OLS().fit(X_, y_).solution

        return self


class DAIVProjectedLeastSquares(DAIVRegressor):
    def fit(self, X, y, G=None, GX=None):
        erm = OLS().fit(GX, y).solution

        s1 = OLS().fit(G, GX).solution
        GXhat = G @ s1

        s2 = erm - np.linalg.pinv(GXhat) @ (GXhat @ erm - y)
        self._W = s2

        return self

