import numpy as np

from src.regressors.abstract import DAIVRegressor
from src.regressors.erm import LeastSquaresClosedForm as OLS

from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV


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


class DAIVGeneralizedMomentMethod(IV, ERM):
    def __init__(self,
                 model="linear",
                 alpha=1.0,
                 gmm_steps=20,
                 epochs=200):
        self.alpha = alpha
        self.model = model  # TODO: refactor code
        super(DAIVGeneralizedMomentMethod,
                     self).__init__(model=model,
                                    gmm_steps=gmm_steps,
                                    epochs=epochs)

    def fit(self, X, y, G, GX):
        return super(DAIVGeneralizedMomentMethod,
                     self).fit(X=GX, y=y, Z=G)
    
    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @property
    def gmm_steps(self):
        return self._gmm_steps

    @gmm_steps.setter
    def gmm_steps(self, gmm_steps):
        self._gmm_steps = gmm_steps
    
    def loss(self,
             X, y, G,
             weights):
        loss = ERM._loss(X, y, self.f) + \
                        self.alpha * IV._loss(X, y, G, self.f, weights)
        self._optimizer.zero_grad()
        loss.backward()
        return loss

