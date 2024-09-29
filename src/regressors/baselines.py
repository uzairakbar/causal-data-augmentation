import torch
import numpy as np
import cvxpy as cp
from typing import List
from loguru import logger
from torch import autograd
from torch.autograd import grad
import torch.nn.functional as F
from scipy.stats import ttest_ind
from scipy.stats import f as fdist
import torch.utils.data as data_utils
from itertools import chain, combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.regressors.abstract import BaselineRegressor

from src.regressors.utils import MODELS, Model, device


DEVICE: str=device()
MAX_BATCH: int=1_000
LOG_FREQUENCY: int=100


class LinearDRO(BaselineRegressor):
    def __init__(self):
        super(LinearDRO, self).__init__()
        
    def _fit(
            self,
            X, y, Z,
            lr: float=0.001,
            epochs: int=40,
            **kwargs
        ):
        logger.info(f'Fitting {self.__class__.__name__} with lr={lr}.')
        N, M = X.shape
        X, y = (
            torch.tensor(X,  dtype=torch.float32).to(DEVICE),
            torch.tensor(y,  dtype=torch.float32).to(DEVICE)
        )
        self.w, self.phi = torch.ones(M, 1), torch.eye(M, M)
        self.w, self.phi = self.w.to(DEVICE), self.phi.to(DEVICE)
        self.w = torch.nn.Parameter(self.w, requires_grad=True)
        self.phi = torch.nn.Parameter(self.phi, requires_grad=True)
        
        opt = torch.optim.Adam([self.phi, self.w], lr=lr)
        mse = torch.nn.MSELoss(reduction='mean')

        environments = np.unique(Z, axis=0)

        for iteration in range(epochs):
            max_error = 0
            for e in range(len(environments)):
                environment = environments[e, :]
                idx_e = np.where((Z == environment).all(axis=1))[0]

                x_e, y_e = X[idx_e, :], y[idx_e, :]
                
                error_e = mse(x_e @ self.phi @ self.w, y_e)
                if error_e > max_error:
                    max_error = error_e
            
            opt.zero_grad()
            max_error.backward()
            opt.step()
            if not iteration%LOG_FREQUENCY:
                logger.info(
                    f'Iteration {iteration}/{epochs} -- Loss: {max_error.item():.2f}.'
                )
        
        self.w = self.w.cpu().detach().numpy()
        self.phi = self.phi.cpu().detach().numpy()
        self._W = (self.phi @ self.w).reshape(-1, 1)
        return self
    
    def _predict(self, X):
        return X @ self._W


class LinearMinimaxREx(BaselineRegressor):
    def __init__(self, alpha: float=0.0, **kwargs):
        super(LinearMinimaxREx, self).__init__(alpha=alpha)
        
    def _fit(
            self,
            X, y, Z,
            lr: float=0.001,
            epochs: int=40,
            **kwargs
        ):
        logger.info(f'Fitting {self.__class__.__name__} with lr={lr}.')
        N, M = X.shape
        X, y = (
            torch.tensor(X,  dtype=torch.float32).to(DEVICE),
            torch.tensor(y,  dtype=torch.float32).to(DEVICE)
        )
        self.w, self.phi = torch.ones(M, 1), torch.eye(M, M)
        self.w, self.phi = self.w.to(DEVICE), self.phi.to(DEVICE)
        self.w = torch.nn.Parameter(self.w, requires_grad=True)
        self.phi = torch.nn.Parameter(self.phi, requires_grad=True)
        
        opt = torch.optim.Adam([self.phi, self.w], lr=lr)
        mse = torch.nn.MSELoss(reduction='mean')

        environments = np.unique(Z, axis=0)
        m = len(environments)
        for iteration in range(epochs):
            max_error = 0
            sum_error = 0
            for e in range(m):
                environment = environments[e, :]
                idx_e = np.where((Z == environment).all(axis=1))[0]

                x_e, y_e = X[idx_e, :], y[idx_e, :]
                
                error_e = mse(x_e @ self.phi @ self.w, y_e)
                sum_error += error_e
                if error_e > max_error:
                    max_error = error_e
            
            opt.zero_grad()
            loss = (
                (1 - m * self._alpha) * max_error + self._alpha * sum_error
            )
            loss.backward()
            opt.step()
            if not iteration%LOG_FREQUENCY:
                logger.info(
                    f'Iteration {iteration}/{epochs} -- Loss: {loss.item():.2f}.'
                )
        
        self.w = self.w.cpu().detach().numpy()
        self.phi = self.phi.cpu().detach().numpy()
        self._W = (self.phi @ self.w).reshape(-1, 1)
        return self
    
    def _predict(self, X):
        return X @ self._W


class LinearVarianceREx(BaselineRegressor):
    def __init__(self, alpha: float=1.0, **kwargs):
        super(LinearVarianceREx, self).__init__(alpha=alpha)
        
    def _fit(
            self,
            X, y, Z,
            lr: float=0.001,
            epochs: int=40,
            **kwargs
        ):
        logger.info(f'Fitting {self.__class__.__name__} with lr={lr}.')
        N, M = X.shape
        X, y = (
            torch.tensor(X,  dtype=torch.float32).to(DEVICE),
            torch.tensor(y,  dtype=torch.float32).to(DEVICE)
        )
        self.w, self.phi = torch.ones(M, 1), torch.eye(M, M)
        self.w, self.phi = self.w.to(DEVICE), self.phi.to(DEVICE)
        self.w = torch.nn.Parameter(self.w, requires_grad=True)
        self.phi = torch.nn.Parameter(self.phi, requires_grad=True)
        
        opt = torch.optim.Adam([self.phi, self.w], lr=lr)
        mse = torch.nn.MSELoss(reduction='mean')

        environments = np.unique(Z, axis=0)
        m = len(environments)
        for iteration in range(epochs):
            errors = []
            for e in range(m):
                environment = environments[e, :]
                idx_e = np.where((Z == environment).all(axis=1))[0]

                x_e, y_e = X[idx_e, :], y[idx_e, :]
                
                error_e = mse(x_e @ self.phi @ self.w, y_e)
                errors.append(error_e)
            
            errors = torch.stack(errors)
            
            opt.zero_grad()
            loss = (
                self._alpha * torch.var(errors) + torch.sum(errors)
            )
            loss.backward()
            opt.step()
            if not iteration%LOG_FREQUENCY:
                logger.info(
                    f'Iteration {iteration}/{epochs} -- Loss: {loss.item():.2f}.'
                )
        
        self.w = self.w.cpu().detach().numpy()
        self.phi = self.phi.cpu().detach().numpy()
        self._W = (self.phi @ self.w).reshape(-1, 1)
        return self
    
    def _predict(self, X):
        return X @ self._W


class LinearRICE(BaselineRegressor):
    def __init__(self, alpha: float=1.0, **kwargs):
        super(LinearRICE, self).__init__(alpha=alpha)
        
    def _fit(
            self,
            X, y,
            da,
            num_augmentations: int=2,
            lr: float=0.001,
            epochs: int=40,
            **kwargs
        ):
        logger.info(f'Fitting {self.__class__.__name__} with lr={lr}.')
        X, _, y, _ = train_test_split(
            X, y, train_size=1.0/num_augmentations
        )
        I_g = ([
            torch.tensor(da(X)[0],  dtype=torch.float32).to(DEVICE)
            for _ in range(num_augmentations)
        ])

        N, M = X.shape
        X, y = (
            torch.tensor(X,  dtype=torch.float32).to(DEVICE),
            torch.tensor(y,  dtype=torch.float32).to(DEVICE)
        )

        self.w, self.phi = torch.ones(M, 1), torch.eye(M, M)
        self.w, self.phi = self.w.to(DEVICE), self.phi.to(DEVICE)
        self.w = torch.nn.Parameter(self.w, requires_grad=True)
        self.phi = torch.nn.Parameter(self.phi, requires_grad=True)
        
        opt = torch.optim.Adam([self.phi, self.w], lr=lr)
        mse = torch.nn.MSELoss(reduction='mean')
        D = torch.nn.MSELoss(reduction='none')

        for iteration in range(epochs):
            penalties_T = []
            for TX in I_g:
                penalty_T = D(
                    X @ self.phi @ self.w, TX @ self.phi @ self.w
                )
                penalties_T.append(penalty_T)
            
            penalties_T = torch.stack(penalties_T)
            sup_T_penalty = torch.max(penalties_T, axis=0)[0]
            
            opt.zero_grad()
            loss = (
                mse(X @ self.phi @ self.w, y) + self._alpha * torch.mean(sup_T_penalty)
            )
            loss.backward()
            opt.step()
            if not iteration%LOG_FREQUENCY:
                logger.info(
                    f'Iteration {iteration}/{epochs} -- Loss: {loss.item():.2f}.'
                )
        
        self.w = self.w.cpu().detach().numpy()
        self.phi = self.phi.cpu().detach().numpy()
        self._W = (self.phi @ self.w).reshape(-1, 1)
        return self
    
    def _predict(self, X):
        return X @ self._W


class LinearAnchorRegression(BaselineRegressor):
    def __init__(self, alpha: float=1.0):
        super(LinearAnchorRegression, self).__init__(alpha=alpha)
        
    def _fit(
            self,
            X, y, Z,
            **kwargs
        ):
        N = len(X)
        I = np.eye(N)
        Cgg = Z.T @ Z
        PI_Z = Z @ np.linalg.pinv( Cgg ) @ Z.T
        
        K = (I + (np.sqrt(self._alpha) - 1) * PI_Z)
        X_, y_ = K @ X, K @ y
        
        self._W = LinearRegression(
            fit_intercept=False
        ).fit(X_, y_).coef_.reshape(-1, 1)

        return self
    
    def _predict(self, X):
        return X @ self._W


class LinearIRM(BaselineRegressor):
    def __init__(self, alpha: float=0.001):
        super(LinearIRM, self).__init__(alpha)
        
    def _fit(
            self,
            X, y, Z,
            lr: float=0.001,
            epochs: int=40,
            **kwargs
        ):
        logger.info(f'Fitting {self.__class__.__name__} with lr={lr}.')
        N, M = X.shape
        X, y = (
            torch.tensor(X,  dtype=torch.float32).to(DEVICE),
            torch.tensor(y,  dtype=torch.float32).to(DEVICE)
        )
        self.phi = torch.eye(M, M)
        self.w = torch.ones(M, 1)
        self.phi, self.w = self.phi.to(DEVICE), self.w.to(DEVICE)
        self.phi = torch.nn.Parameter(self.phi)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=lr)
        mse = torch.nn.MSELoss()

        environments = np.unique(Z, axis=0)

        for iteration in range(epochs):
            penalty = 0
            error = 0
            for e in range(len(environments)):
                environment = environments[e, :]
                idx_e = np.where((Z == environment).all(axis=1))[0]

                # if len(idx_e) == 1:
                #     continue

                x_e, y_e = X[idx_e, :], y[idx_e, :]
                
                error_e = mse(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e
            
            # logger.info(f'IRM iteration {iteration}')

            opt.zero_grad()
            loss = (self._alpha * error + (1 - self._alpha) * penalty)
            loss.backward()
            opt.step()
            if not iteration%LOG_FREQUENCY:
                logger.info(
                    f'Iteration {iteration}/{epochs} -- Loss: {loss.item():.2f}.'
                )
        
        self.w = self.w.cpu().detach().numpy()
        self.phi = self.phi.cpu().detach().numpy()
        self._W = (self.phi @ self.w).reshape(-1, 1)
        return self
    
    def _predict(self, X):
        return X @ self._W


class InvariantCausalPrediction(BaselineRegressor):
    def __init__(self, alpha: float=0.05):
        super(InvariantCausalPrediction, self).__init__(alpha)
        
    def _fit(
            self,
            X, y, Z,
            verbose: bool=True,
            **kwargs
        ):
        N, M = X.shape
        environments = np.unique(Z, axis=0)

        accepted_subsets = []
        for subset in self.powerset(range(M)):
            if len(subset) == 0:
                continue

            x_s = X[:, subset]
            reg = LinearRegression(fit_intercept=False).fit(x_s, y)

            p_values = []
            for e in range(len(environments)):
                environment = environments[e, :]
                e_in = np.where((Z == environment).all(axis=1))[0]
                e_out = np.where((Z != environment).any(axis=1))[0]

                if len(e_in) == 1:
                    continue

                total = len(e_in) + len(e_out)
                if total != len(Z):
                    logger.info(f'e_in + e_out = {total} != {len(Z)}')
                
                res_in = (y[e_in, :] - reg.predict(x_s[e_in, :])).ravel()
                res_out = (y[e_out, :] - reg.predict(x_s[e_out, :])).ravel()

                p_values.append(self.mean_var_test(res_in, res_out))
            
            # TODO: Jonas uses "min(p_values) * len(environments) - 1"
            p_value = min(p_values) * len(environments)

            if p_value > self._alpha:
                accepted_subsets.append(set(subset))
                logger.info(f'Accepted subset: {subset}')
        
        if len(accepted_subsets):
            accepted_features = list(set.intersection(*accepted_subsets))
            logger.info(f'Intersection: {accepted_features}')
            self.coefficients = np.zeros(M)

            if len(accepted_features):
                x_s = X[:, list(accepted_features)]
                reg = LinearRegression(fit_intercept=False).fit(x_s, y)
                self.coefficients[list(accepted_features)] = reg.coef_
        else:
            self.coefficients = np.zeros(M)
        
        self._W = self.coefficients.reshape(-1, 1)
        return self

    def _predict(self, X):
        return X @ self._W

    def mean_var_test(self, X, y):
        pvalue_mean = ttest_ind(X, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(X, ddof=1) / np.var(y, ddof=1),
                                    X.shape[0] - 1,
                                    y.shape[0] - 1)
        
        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)

    def powerset(self, s: List[int]):
        return chain.from_iterable(
            combinations(s, r) for r in range(len(s) + 1)
        )


class InvariantRiskMinimization(BaselineRegressor):
    all_models = MODELS

    def __init__(self, model: Model='linear', alpha: float=0.001):
        if model in self.all_models:
            self.model = self.all_models[model]
        else:
            raise ValueError(f'model has invalid value {str(model)}')
        self.optimizer = None
        super(InvariantRiskMinimization, self).__init__(alpha)

    def fit_f_minibatch(self, train):
        losses = []
        for i, (X_b, y_b, Z_b) in enumerate(train):
            X_b, y_b, Z_b = (
                X_b.to(DEVICE), y_b.to(DEVICE), Z_b.to(DEVICE)
            )
            loss_val = self._optimizer.step(
                lambda: self.loss(X_b, y_b, Z_b)
            )
            losses += [loss_val.data.cpu().numpy()]
        logger.info(f'  train loss {np.mean(losses):.2f}')

    def fit_f_batch(self, X, y, Z):
        _ = self._optimizer.step(lambda: self.loss(X, y, Z))
    
    def _fit(
            self,
            X, y, Z,
            lr: float=0.001, epochs: int=40, batch: int=128,
            pbar_manager=None,
            **kwargs
        ):

        n, m = X.shape
        _, k = Z.shape

        self.f = self.model(m)
        self.f.float()
        self.f.train()
        self.f = self.f.to(DEVICE)
        
        self._optimizer = torch.optim.Adam(self.f.parameters(), lr=lr)
        
        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.float)
        X = torch.tensor(X, dtype=torch.float)
        Z = torch.tensor(Z, dtype=torch.float)
        
        
        X, y, Z = (
            X.to(DEVICE), y.to(DEVICE), Z.to(DEVICE)
        )

        batch_mode = 'mini' if n >= MAX_BATCH else 'full'
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y, Z),
                                      batch_size=batch, shuffle=True)
        
        if pbar_manager:
            method_name = self.__class__.__name__
            pbar_epochs = pbar_manager.counter(
                total=epochs, desc=f'{method_name}', unit='epochs', leave=False
            )
        for epoch in range(epochs):
            if batch_mode == 'full':
                self.fit_f_batch(X, y)
            else:
                # logger.info(f'g epoch {epoch + 1}/{epochs}')
                self.fit_f_minibatch(train)
            
            pbar_epochs.update()
        pbar_epochs.close()
        self.f.eval()
        return self

    def _predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        X, _ = X.to(DEVICE)
        
        output = self.f(X).data.cpu().numpy()
        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            output = output.argmax(axis=1)[:, None].astype(int)
        elif isinstance(self.f[-1], torch.nn.Sigmoid):
            output = (output > 0.5).astype(int)
        return output
    
    @property
    def solution(self):
        W = []
        for w in self.f.parameters():
            W.append(w.view(-1, 1))
        W = torch.cat(W)
        return W.data.cpu().numpy()
    
    def loss(self,
             X, y, Z):
        loss = self._loss(X, y, Z, self.f, self._alpha)
        self._optimizer.zero_grad()
        loss.backward()
        return loss

    @classmethod
    def _loss(cls,
              X, y, Z, f,
              alpha):
        if not isinstance(f[-1], torch.nn.Linear):
            logits = f[:-1](X)
        else:
            logits = f(X)
        
        env_losses = []
        environments = torch.unique(Z, dim=0)
        for e in range(len(environments)):
            environment = environments[e, :]
            e_idx = (Z == environment).all(dim=1).nonzero(as_tuple=False)[:, 0]

            env_losses.append(
                cls._env_loss(
                    X[e_idx, :], logits[e_idx, :], y[e_idx, :], alpha
                )
            )
        
        loss = torch.stack(env_losses).mean()

        return loss
    
    @staticmethod
    def _env_loss(X, logits, y, alpha):
        
        def mean_nll(logits, y):
            return F.binary_cross_entropy_with_logits(
                logits, y, reduction='mean'
            )
        
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        gradient = autograd.grad(loss, [scale], create_graph=True)[0]
        penalty = torch.sum(gradient**2)

        loss = mean_nll(logits, y)

        return loss + alpha * penalty
