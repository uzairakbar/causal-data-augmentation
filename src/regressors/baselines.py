import torch
import random
import numpy as np
from typing import List
from loguru import logger
from torch import autograd
from abc import abstractmethod
import torch.nn.functional as F
import torch.utils.data as data_utils
from itertools import chain, combinations
from scipy.stats import f as fdist, ttest_ind
from sklearn.linear_model import LinearRegression

from src.regressors.abstract import BaselineRegressor

from src.regressors.utils import MODELS, Model, device


DEVICE: str=device()
MAX_BATCH: int=2_500
LOG_FREQUENCY: int=100
MAX_ICP_SUBSETS: int=1_024


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


class InvariantCausalPrediction(BaselineRegressor):
    def __init__(self, model='linear', alpha: float=0.05):
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
        randomised_subsets = self._random_powerset(range(M))
        for i, subset in enumerate(randomised_subsets):
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

                p_values.append(self._mean_var_test(res_in, res_out))
            
            # Jonas uses `min(p_values) * len(environments) - 1`
            p_value = min(p_values) * len(environments)

            if p_value > self._alpha:
                accepted_subsets.append(set(subset))
                logger.info(f'Accepted subset: {subset}')
            
            if i >= MAX_ICP_SUBSETS:
                break
        
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

    def _mean_var_test(self, X, y):
        pvalue_mean = ttest_ind(X, y, equal_var=False).pvalue
        pvalue_var1 = 1 - fdist.cdf(np.var(X, ddof=1) / np.var(y, ddof=1),
                                    X.shape[0] - 1,
                                    y.shape[0] - 1)
        
        pvalue_var2 = 2 * min(pvalue_var1, 1 - pvalue_var1)

        return 2 * min(pvalue_mean, pvalue_var2)
    
    def _random_powerset(self, s: List[int]):
        '''
        Since exhaustive search over all feature combinations can be expensive,
        we randomise the powerset for reasonable exploration of feature combos 
        and only try the first `MAX_ICP_SUBSETS` (non-singleton) subsets.
        '''
        s = list(s)
        lengths = list(range(len(s)+1))
        random.shuffle(lengths)
        return chain.from_iterable(
            combinations(s, r) for r in lengths if not random.shuffle(s)
        )


class NonlinearBaselineRegressor(BaselineRegressor):
    all_models = MODELS

    def __init__(self, model: Model='linear', alpha: float=0.001):
        super(NonlinearBaselineRegressor, self).__init__(alpha)
        if model in self.all_models:
            self._model = self.all_models[model]
        else:
            raise ValueError(f'model has invalid value {str(model)}')
        self.optimizer = None

    def fit_f_minibatch(self, train):
        losses = []
        for i, (X_b, y_b, Z_b) in enumerate(train):
            loss_val = self._optimizer.step(
                lambda: self.loss(X_b, y_b, Z_b)
            )
            losses += [loss_val.data.cpu().numpy()]

    def fit_f_batch(self, X, y, Z):
        loss = self._optimizer.step(lambda: self.loss(X, y, Z))
    
    def _fit(
            self,
            X, y, Z,
            lr: float=0.001, epochs: int=40, batch: int=128,
            pbar_manager=None,
            **kwargs
        ):
        n, m = X.shape
        _, k = Z.shape

        self.f = self._model(m)
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

        batch_mode = 'mini' if n > MAX_BATCH else 'full'
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y, Z),
                                      batch_size=batch, shuffle=True)
        
        method_name = self.__class__.__name__
        logger.info(
            f'Training {method_name} in {batch_mode}-batch mode with lr={lr}, epoch={epochs}, batch={batch}'
        )
        if pbar_manager:
            pbar_epochs = pbar_manager.counter(
                total=epochs, desc=f'{method_name}', unit='epochs', leave=False
            )
        for epoch in range(epochs):
            if batch_mode == 'full':
                self.fit_f_batch(X, y, Z)
            else:
                # logger.info(f'g epoch {epoch + 1}/{epochs}')
                self.fit_f_minibatch(train)
            
            if pbar_manager: pbar_epochs.update()
        if pbar_manager: pbar_epochs.close()
        self.f.eval()
        return self

    def _predict(self, X):
        X = torch.tensor(X, dtype=torch.float).to(DEVICE)        
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
    @abstractmethod
    def _loss(cls,
              X, y, Z, f,
              alpha):
        pass

    @staticmethod
    def _erm_loss(X, y, f, reduction='mean'):
        y_hat = f(X)
        if isinstance(f[-1], torch.nn.LogSoftmax):
            loss = F.nll_loss(y_hat, y.flatten(), reduction=reduction)
        elif isinstance(f[-1], torch.nn.Sigmoid):
            loss = F.binary_cross_entropy(y_hat, y, reduction=reduction)
        else:
            loss = F.mse_loss(y_hat, y, reduction=reduction)
        return loss


class InvariantRiskMinimization(NonlinearBaselineRegressor):
    def __init__(self, model: Model='linear', alpha: float=0.001, **kwargs):
        self.model, self.alpha = model, alpha
        super(InvariantRiskMinimization, self).__init__(model=model, alpha=alpha)
    
    @classmethod
    def _loss(cls,
              X, y, Z, f,
              alpha):
        
        if isinstance(f[-1], torch.nn.LogSoftmax):
            M = f[-2].out_features
            phi = lambda x: f[:-1](x)
            R = lambda logits, y: F.nll_loss(
                F.log_softmax(logits, dim=1), y.flatten(), reduction='mean'
            )
        elif isinstance(f[-1], torch.nn.Sigmoid):
            M = 1
            phi = lambda x: f[:-1](x)
            R = lambda logits, y: F.binary_cross_entropy_with_logits(
                logits, y, reduction='mean'
            )
        elif isinstance(f[-1], torch.nn.Linear):
            M = f[-1].out_features
            phi = lambda x: f(x)
            R = lambda y_hat, y: F.mse_loss(
                y_hat, y, reduction='mean'
            )
        
        w = torch.ones(M, 1).to(DEVICE)
        w.requires_grad = True
        
        env_losses = []
        environments = torch.unique(Z, dim=0)
        for e in range(len(environments)):
            environment = environments[e, :]
            e_idx = (Z == environment).all(dim=1).nonzero(as_tuple=False)[:, 0]

            env_losses.append(
                cls._irm_loss(
                    X[e_idx, :], y[e_idx, :], phi, w, R, alpha
                )
            )
        
        loss = torch.stack(env_losses).mean()

        return loss
    
    @staticmethod
    def _irm_loss(X, y, phi, w, R, alpha):

        loss = R(phi(X) @ w, y)
        gradient = autograd.grad(loss, [w], create_graph=True)[0]
        penalty = torch.sum(gradient**2)

        loss = R(phi(X), y)

        return loss + alpha * penalty


class AnchorRegression(NonlinearBaselineRegressor):
    def __init__(self, model: Model='linear', alpha: float=1.0, **kwargs):
        self.model, self.alpha = model, alpha
        super(AnchorRegression, self).__init__(model=model, alpha=alpha)
    
    @classmethod
    def _loss(cls,
              X, y, Z, f,
              alpha):
        N, M = X.shape
        I = torch.eye(N).to(DEVICE)
        Pi = Z @ torch.linalg.pinv( Z.t() @ Z ) @ Z.t()

        mse = F.mse_loss(f(X), y, reduction='none')
        backdoor_adjustment = (I - Pi) @ mse
        iv_regression = Pi @ mse
        
        loss = (backdoor_adjustment + alpha * iv_regression).mean()

        return loss


class VarianceREx(NonlinearBaselineRegressor):
    def __init__(self, model: Model='linear', alpha: float=1.0, **kwargs):
        self.model, self.alpha = model, alpha
        super(VarianceREx, self).__init__(model=model, alpha=alpha)

    @classmethod
    def _loss(cls,
              X, y, Z, f,
              alpha):
        env_losses = []
        environments = torch.unique(Z, dim=0)
        for e in range(len(environments)):
            environment = environments[e, :]
            e_idx = (Z == environment).all(dim=1).nonzero(as_tuple=False)[:, 0]

            env_losses.append(
                cls._erm_loss(
                    X[e_idx, :], y[e_idx, :], f
                )
            )
        
        errors = torch.stack(env_losses)
        
        loss = (
            alpha * torch.var(errors) + torch.sum(errors)
        )

        return loss


class MiniMaxREx(NonlinearBaselineRegressor):
    def __init__(self, model: Model='linear', alpha: float=0.0, **kwargs):
        self.model, self.alpha = model, alpha
        super(MiniMaxREx, self).__init__(model=model, alpha=alpha)
    
    @classmethod
    def _loss(cls,
              X, y, Z, f,
              alpha):
        
        sum_error = 0.0
        max_error = 0.0
        environments = torch.unique(Z, dim=0)

        m = len(environments)
        for e in range(m):
            environment = environments[e, :]
            e_idx = (Z == environment).all(dim=1).nonzero(as_tuple=False)[:, 0]

            error_e = cls._erm_loss(
                X[e_idx, :], y[e_idx, :], f
            )
            sum_error += error_e
            if error_e > max_error:
                max_error = error_e
            
        loss = (
            (1 - m * alpha) * max_error + alpha * sum_error
        )
        return loss


class DistributionallyRobustOptimization(NonlinearBaselineRegressor):
    def __init__(self, model: Model='linear'):
        self.model = model
        super(DistributionallyRobustOptimization, self).__init__(
            model=model, alpha=0.0
        )
    
    @classmethod
    def _loss(cls,
              X, y, Z, f,
              alpha):
        
        max_error = 0.0
        environments = torch.unique(Z, dim=0)
        for e in range(len(environments)):
            environment = environments[e, :]
            e_idx = (Z == environment).all(dim=1).nonzero(as_tuple=False)[:, 0]

            error_e = cls._erm_loss(
                X[e_idx, :], y[e_idx, :], f
            )
            if error_e > max_error:
                max_error = error_e
        
        return max_error


class RICE(NonlinearBaselineRegressor):
    def __init__(self, model: Model='linear', alpha: float=1.0, **kwargs):
        self.model, self.alpha = model, alpha
        super(RICE, self).__init__(model=model, alpha=alpha)
    
    def fit_f_minibatch(self, train):
        losses = []
        for i, (X_b, y_b, *Z_b) in enumerate(train):
            loss_val = self._optimizer.step(
                lambda: self.loss(X_b, y_b, Z_b)
            )
            losses += [loss_val.data.cpu().numpy()]
        # logger.info(f'  mini-batch loss {np.mean(losses):.2f}')
    
    def _fit(
            self,
            X, y,
            da,
            num_augmentations: int=3,
            lr: float=0.001, epochs: int=40, batch: int=128,
            pbar_manager=None,
            **kwargs
        ):
        def flatten(x):
            return x.reshape(*x.shape[:1], -1)

        I_g = ([
            flatten(da(X)[0])
            for _ in range(num_augmentations)
        ])
        I_g = ([
            torch.tensor(GX, dtype=torch.float32).to(DEVICE)
            for GX in I_g
        ])

        X = flatten(X)
        X = torch.tensor(X, dtype=torch.float).to(DEVICE)
        N, M = X.shape

        self.f = self._model(M)
        self.f.float()
        self.f.train()
        self.f = self.f.to(DEVICE)

        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            y = torch.tensor(y, dtype=torch.long).to(DEVICE)
        else:
            y = torch.tensor(y, dtype=torch.float).to(DEVICE)
        
        
        self._optimizer = torch.optim.Adam(self.f.parameters(), lr=lr)
        
        batch_mode = 'mini' if N > MAX_BATCH else 'full'
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y, *I_g),
                                      batch_size=batch, shuffle=True)
        
        method_name = self.__class__.__name__
        logger.info(
            f'Training {method_name} in {batch_mode}-batch mode with lr={lr}, epoch={epochs}, batch={batch}'
        )
        if pbar_manager:
            pbar_epochs = pbar_manager.counter(
                total=epochs, desc=f'{method_name}', unit='epochs', leave=False
            )
        for epoch in range(epochs):
            if batch_mode == 'full':
                self.fit_f_batch(X, y, I_g)
            else:
                # logger.info(f'g epoch {epoch + 1}/{epochs}')
                self.fit_f_minibatch(train)
            
            if pbar_manager: pbar_epochs.update()
        if pbar_manager: pbar_epochs.close()
        self.f.eval()
        return self

    @classmethod
    def _loss(cls,
              X, y, I_g, f,
              alpha):
        D = torch.nn.MSELoss(reduction='none')
        penalties_T = []
        for TX in I_g:
            penalty_T = D(
                f(X), f(TX)
            )
            penalties_T.append(penalty_T)
        
        penalties_T = torch.stack(penalties_T)
        sup_T_penalty = torch.max(penalties_T, axis=0)[0]
        
        loss = (
            cls._erm_loss( X, y, f ) + alpha * torch.mean(sup_T_penalty)
        )
        return loss

    @staticmethod
    def _erm_loss(X, y, f, reduction='mean'):
        y_hat = f(X)
        if isinstance(f[-1], torch.nn.LogSoftmax):
            loss = F.nll_loss(y_hat, y.flatten(), reduction=reduction)
        elif isinstance(f[-1], torch.nn.Sigmoid):
            loss = F.binary_cross_entropy(y_hat, y, reduction=reduction)
        else:
            loss = F.mse_loss(y_hat, y, reduction=reduction)
        return loss
