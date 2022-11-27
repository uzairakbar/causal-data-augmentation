import torch
import numpy as np
import torch.nn.functional as F

from src.regressors.abstract import DAIVRegressor
from src.regressors.erm import LeastSquaresClosedForm as OLS

from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV


class DAIVLeastSquaresClosedForm(DAIVRegressor):
    def __init__(self, alpha = 1.0):
        super(DAIVLeastSquaresClosedForm, self).__init__(alpha)
        
    def _fit(self, X, y, G=None, GX=None):
        N = len(GX)
        I = np.eye(N)
        Cgg = G.T @ G
        PI_G = G @ np.linalg.inv( Cgg ) @ G.T
        
        K = (I + np.sqrt(self._alpha) * PI_G)
        X_, y_ = K @ GX, K @ y
        
        self._W = OLS().fit(X_, y_).solution

        return self
    
    def _predict(self, X):
        return X @ self._W


class DAIVProjectedLeastSquares(DAIVRegressor):
    def _fit(self, X, y, G=None, GX=None):
        erm = OLS().fit(GX, y).solution

        s1 = OLS().fit(G, GX).solution
        GXhat = G @ s1

        s2 = erm - np.linalg.pinv(GXhat) @ (GXhat @ erm - y)
        self._W = s2

        return self
    
    def _predict(self, X):
        return X @ self._W


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


class MinMaxDAIV(IV, ERM):
    def __init__(self,
                 model="linear",
                 epochs=200):
        self.alpha = None
        self.model = model  # TODO: refactor code
        super(MinMaxDAIV,
                     self).__init__(model=model,
                                    gmm_steps=1,
                                    epochs=epochs)
    
    def fit(self, X, y, G, GX):
        _, k = G.shape
        l = 10 if (self.model == "rmnist") else 1
        self.alpha = torch.nn.Linear(k*l, 1, bias=False)
        self._alpha_optimizer = torch.optim.Adam(
            self.alpha.parameters(), lr=0.01, maximize=True
        )
        return super(MinMaxDAIV,
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
        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            y_hat = F.softmax(self.f[:-1](X), dim=1)
            y_onehot = F.one_hot(y.flatten(), num_classes=10)
            mom = self._moment_conditions(
                G, y_onehot, y_hat
            )
        else:
            y_hat = self.f(X)
            mom = self._moment_conditions(G, y, y_hat)
        
        loss = ( ERM._loss(X, y, self.f, reduction="none") - self.alpha(mom) ).sum()

        self._alpha_optimizer.zero_grad()
        self._optimizer.zero_grad()
        loss.backward()
        self._alpha_optimizer.step()
        return loss
    
    def calculate_weights(self):
        # TODO: more speed
        pass