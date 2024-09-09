import torch
import numpy as np
import cvxpy as cp
from math import comb
import torch.nn.functional as F
from scipy.optimize import minimize

from src.regressors.abstract import DAIVRegressor
from src.regressors.erm import LeastSquaresClosedForm as OLS

from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV


class DAIVLeastSquaresClosedForm(DAIVRegressor):
    def __init__(self, alpha = 1.0):
        super(DAIVLeastSquaresClosedForm, self).__init__(alpha)
        
    def _fit(self, X, y, G=None, GX=None, **kwargs):
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


class DAIVProjectedLeastSquaresClosedForm(DAIVRegressor):
    def __init__(self, alpha = None):
        super(DAIVProjectedLeastSquaresClosedForm, self).__init__(alpha)

    def _fit(self, X, y, G=None, GX=None):
        h_erm = OLS().fit(GX, y).solution

        s1 = OLS().fit(G, GX).solution
        GXhat = G @ s1
        
        A = GXhat.T @ GXhat
        b = GXhat.T @ y
        
        def f(h):
            return 0.5 * np.sum( (h_erm - h[:, None])**2 )

        def df(h):
            return - 1.0 * (h_erm - h[:, None])
        
        def ddf(h):
            return np.eye(len(h_erm))

        constraints = (
            {'type': 'eq',
            'fun' : lambda h: A @ h[:, None],
            'jac' : lambda h: A,},
            {'type': 'ineq',
            'fun' : lambda h: A @ h[:, None],
            'jac' : lambda h: A,},
            {'type': 'ineq',
            'fun' : lambda h: - A @ h[:, None],
            'jac' : lambda h: - A,},
        )

        h0 = h_erm - A.T @ np.linalg.pinv(A @ A.T) @ (A @ h_erm - b)
        self._W = h0

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

        def f(h):
            return 0.5 * np.sum( (h_erm - h[:, None])**2 )

        def df(h):
            return - 1.0 * (h_erm - h[:, None])
        
        def ddf(h):
            return np.eye(len(h_erm))

        constraints = (
            {'type': 'eq',
            'fun' : lambda h: b - A @ h[:, None],
            'jac' : lambda h: -A,},
            {'type': 'ineq',
            'fun' : lambda h: b - A @ h[:, None],
            'jac' : lambda h: -A,},
            {'type': 'ineq',
            'fun' : lambda h: A @ h[:, None] - b,
            'jac' : lambda h: A,},
        )

        h0 = h_erm - A.T @ np.linalg.pinv(A @ A.T) @ (A @ h_erm - b)
        # result = minimize(
        #     f, h0, method='SLSQP', jac=df, hess=ddf,
        #     constraints=constraints,
        #     options={'ftol': 1e-8, 'disp': False, }#'eps':0.001},
        # )
        # self._W = result.x[:, None]



        h = cp.Variable(h_erm.shape)
        cost = cp.norm(h - h_erm)
        constraints = [A@h == b]
        prob = cp.Problem(
            cp.Minimize(cost),
            constraints
        )
        result = prob.solve(solver=cp.ECOS)
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
        result = prob.solve(solver=cp.ECOS)
        self._W = h.value
        return self
    
    def _predict(self, X):
        return X @ self._W


class DAIVGeneralizedMomentMethod(IV, ERM):
    def __init__(self,
                 model='linear',
                 alpha=1.0):
        self.alpha = alpha
        self.model = model  # TODO: refactor code
        super(DAIVGeneralizedMomentMethod,
                     self).__init__(model=model)
    
    def fit(self, X, y, G, GX, **kwargs):
        return super(DAIVGeneralizedMomentMethod,
                     self).fit(X=GX, y=y, Z=G, **kwargs)
    
    def loss(self,
             X, y, G,
             weights):
        loss = ERM._loss(X, y, self.f) + \
                        self.alpha * IV._loss(X, y, G, self.f, weights)
        self._optimizer.zero_grad()
        loss.backward()
        return loss


class MinMaxDAIV(IV, ERM):
    def __init__(self, model='linear'):
        self.alpha = None
        self.model = model  # TODO: refactor code
        super(MinMaxDAIV,
                     self).__init__(model=model)
    
    def fit(self, X, y, G, GX, lr=0.001, **kwargs):
        _, k = G.shape
        G_poly_degree = 2
        alpha_in_dim = comb(k+G_poly_degree, G_poly_degree) - 1
        self.alpha = torch.nn.Linear(alpha_in_dim, 1, bias=False)
        if torch.cuda.is_available():
            self.f = self.alpha.cuda()
        elif torch.backends.mps.is_available():
            self.f = self.alpha.to('mps')
        
        self._alpha_optimizer = torch.optim.Adam(
            self.alpha.parameters(), lr=lr, maximize=True
        )
        
        return super(MinMaxDAIV,
                     self).fit(X=GX, y=y, Z=G, lr=lr, **kwargs)
    
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
        
        # TODO: check which one of these is correct!
        # loss = ( ERM._loss(X, y, self.f, reduction='none') - self.alpha(mom) ).sum()
        mom = mom.mean(dim=0, keepdim=True)
        loss = ERM._loss(X, y, self.f, reduction='mean') - self.alpha(mom)

        self._alpha_optimizer.zero_grad()
        self._optimizer.zero_grad()
        loss.backward()
        self._alpha_optimizer.step()
        return loss
    
    def calculate_weights(self):
        # TODO: more speed
        pass


class DAIVConstrainedOptimizationGMM(IV, ERM):
    def __init__(self, model='linear'):
        self.alpha = None
        self.model = model  # TODO: refactor code
        self.erm = ERM(model=model)
        super(DAIVConstrainedOptimizationGMM,
                     self).__init__(model=model)
    
    def fit(
            self, X, y, G, GX, lr=0.001, batch=512, epochs1=4, epochs2=10, **kwargs
        ):
        _, k = G.shape
        G_poly_degree = 2
        alpha_in_dim = comb(k+G_poly_degree, G_poly_degree) - 1
        self.alpha = torch.nn.Linear(alpha_in_dim, 1, bias=False)
        if torch.cuda.is_available():
            self.f = self.alpha.cuda()
        elif torch.backends.mps.is_available():
            self.f = self.alpha.to('mps')
        
        self._alpha_optimizer = torch.optim.Adam(
            self.alpha.parameters(), lr=lr, maximize=True
        )
        
        self.erm.fit(
            GX, y, lr=lr, epochs=epochs1*epochs2
        )
        return super(DAIVConstrainedOptimizationGMM,
                     self).fit(
                         X=GX, y=y, Z=G,
                         lr=lr, batch=batch, epochs1=epochs1, epochs2=epochs2,
                         **kwargs
                        )

    def loss(self,
             X, y, G,
             weights):
        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            y_erm = F.softmax(self.erm.f[:-1](X), dim=1)

            y_hat = F.softmax(self.f[:-1](X), dim=1)
            y_onehot = F.one_hot(y.flatten(), num_classes=10)
            mom = self._moment_conditions(
                G, y_onehot, y_hat
            )
        else:
            y_erm = self.erm.f(X)

            y_hat = self.f(X)
            mom = self._moment_conditions(G, y, y_hat)
        
        # l2 = sum(
        #     (x - y).abs().sum() 
        #     for x, y in zip(
        #         self.erm.state_dict().values(), self.f.state_dict().values()
        #     )
        # )

        # mom = mom.mean(dim=0, keepdim=True)
        # loss = 0.5*l2 - self.alpha(mom)
        mom = mom.mean(dim=0, keepdim=True)
        loss = ERM._loss(X, y_erm, self.f, reduction='mean') - self.alpha(mom)

        self._alpha_optimizer.zero_grad()
        self._optimizer.zero_grad()
        loss.backward()
        self._alpha_optimizer.step()
        return loss
    
    def calculate_weights(self):
        # TODO: more speed
        pass