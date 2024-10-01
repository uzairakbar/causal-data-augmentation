import torch
import numpy as np
from loguru import logger
import torch.nn.functional as F
import torch.utils.data as data_utils

from src.regressors.abstract import IVRegressor as IV

from src.regressors.erm import (
    LeastSquaresClosedForm as LSCF,
    LeastSquaresGradientDescent as LSGD,
    LeastSquaresCvxpy as LSGD,
)

from src.regressors.utils import MODELS, Model, device


DEVICE: str=device()
MAX_BATCH: int=2_500
LOG_FREQUENCY: int=100
ERM = {
    'cf': LSCF(),
    'gd': LSGD()
}


class IVTwoStageLeastSquares(IV):
    def __init__(self, s1='cf', s2='gd', **kwargs):
        self.s1 = s1
        self.s2 = s2
        super(IVTwoStageLeastSquares, self).__init__(**kwargs)
    
    def _fit(self, X, y, Z, **kwargs):

        S1 = ERM[self.s1].fit(Z, X).solution
        Xhat = Z @ S1

        S2 = ERM[self.s2].fit(Xhat, y).solution
        self._W = S2

        return self
    
    def _predict(self, X):
        return X @ self._W


class IVGeneralizedMomentMethod(IV):
    _models = MODELS

    def __init__(self, model: Model='linear'):
        if model in self._models:
            self.__model = self._models[model]
        else:
            raise ValueError(f'model has invalid value {str(model)}')
        self._optimizer = None
        super(IVGeneralizedMomentMethod, self).__init__()

    def fit_f_minibatch(self, train, weights):
        losses = list()
        for i, (X_b, y_b, Z_b) in enumerate(train):
            X_b, y_b, Z_b, weights = (
                X_b.to(DEVICE), y_b.to(DEVICE), Z_b.to(DEVICE), weights.to(DEVICE)
            )
            loss_val = self._optimizer.step(
                lambda: self.loss(X_b, y_b, Z_b, weights)
            )
            losses += [loss_val.data.cpu().numpy()]
        # logger.info(f'  train loss {np.mean(losses):.2f}')

    def fit_f_batch(self, X, y, Z, weights):
        _ = self._optimizer.step(lambda: self.loss(X, y, Z, weights))

    def _fit(
            self,
            X, y, Z,
            lr=0.001, batch=512, epochs1=4, epochs2=10,
            pbar_manager=None
        ):
        from sklearn.preprocessing import PolynomialFeatures
        Z_poly_degree = 2
        Z = PolynomialFeatures(
            Z_poly_degree, include_bias=False
        ).fit_transform(Z)

        n, m = X.shape
        _, k = Z.shape

        self.f = self.__model(m)
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
        
        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            weights = torch.eye(10*k)
        else:
            weights = torch.eye(k)
        
        X, y, Z, weights = (
            X.to(DEVICE), y.to(DEVICE), Z.to(DEVICE), weights.to(DEVICE)
        )

        batch_mode = 'mini' if n > MAX_BATCH else 'full'
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y, Z),
                                      batch_size=batch, shuffle=True)
        
        method_name = self.__class__.__name__
        logger.info(
            f'Training {method_name} in {batch_mode}-batch mode with lr={lr}, epochs1={epochs1}, epochs2={epochs2}, batch={batch}'
        )
        if pbar_manager:
            pbar_gmm = pbar_manager.counter(
                total=epochs1, desc=f'{method_name}', unit='GMM steps', leave=False
            )
        for step in range(epochs1):
            # logger.info(f'GMM step {step + 1}/{epochs1}')
            if step > 0:
                # optimize weights
                with torch.no_grad():
                    if isinstance(self.f[-1], torch.nn.LogSoftmax):
                        y_hat = F.softmax(self.f[:-1](X), dim=1)
                        y_onehot = F.one_hot(
                            y.flatten(), num_classes=10
                        )
                        moment_conditions = self._moment_conditions(
                            Z, y_onehot, y_hat
                        )
                    else:
                        y_hat = self.f(X)
                        moment_conditions = self._moment_conditions(
                            Z, y, y_hat
                        )
                    covariance_matrix = torch.mm(
                        moment_conditions.t(), moment_conditions
                    ) / n
                    # weights = torch.cholesky_inverse(
                    #     torch.linalg.cholesky(
                    #         covariance_matrix
                    #         + 1e-7*torch.eye(covariance_matrix.size(dim=-1), device='mps')
                    #     )
                    # )
                    weights = torch.as_tensor(
                        np.linalg.pinv(
                            covariance_matrix.cpu().numpy() + 1e-7*np.eye(covariance_matrix.size(dim=-1)),
                            rcond=1e-9
                        ),
                        dtype=torch.float
                    )
                    weights = weights.to(DEVICE)
            
            if pbar_manager:
                pbar_epochs = pbar_manager.counter(
                    total=epochs2, desc=f'Step {step}', unit='epochs', leave=False
                )
            for epoch in range(epochs2):
                if batch_mode == 'full':
                    self.fit_f_batch(X, y, Z, weights)
                else:
                    # logger.info(f'g epoch {epoch + 1}/{epochs2}')
                    self.fit_f_minibatch(train, weights)
                
                if pbar_manager: pbar_epochs.update()
            if pbar_manager: pbar_epochs.close()
            if pbar_manager: pbar_gmm.update()
        if pbar_manager: pbar_gmm.close()

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
             X, y, Z,
             weights):
        loss = self._loss(X, y, Z, self.f, weights)
        self._optimizer.zero_grad()
        loss.backward()
        return loss

    @classmethod
    def _loss(cls,
              X, y, Z, f,
              weights):
        if isinstance(f[-1], torch.nn.LogSoftmax):
            y_hat = F.softmax(f[:-1](X), dim=1)
            y_onehot = F.one_hot(y.flatten(), num_classes=10)
            moment_conditions = cls._moment_conditions(
                Z, y_onehot, y_hat
            )
        else:
            y_hat = f(X)
            moment_conditions = cls._moment_conditions(Z, y, y_hat)
        moms = moment_conditions.mean(dim=0, keepdim=True)
        loss = torch.mm(torch.mm(moms, weights), moms.t())
        return loss
    
    @staticmethod
    def _moment_conditions(Z, y, y_hat):
        moment_conditions = torch.einsum('bi,bj->bij', (Z, y - y_hat))
        moment_conditions = moment_conditions.view(
            moment_conditions.size(0), -1
        )
        return moment_conditions

