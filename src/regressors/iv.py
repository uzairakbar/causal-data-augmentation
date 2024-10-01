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
    all_models = MODELS

    def __init__(self, model: Model='linear'):
        super(IVGeneralizedMomentMethod, self).__init__()
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
        loss = self._loss(X, y, Z, self.f)
        self._optimizer.zero_grad()
        loss.backward()
        return loss

    @classmethod
    def _loss(cls,
              X, y, Z,
              f):
        mse = F.mse_loss(f(X), y, reduction='none')

        gmm_weights = torch.linalg.pinv( Z.t() @ Z )
        Pi = Z @ gmm_weights @ Z.t()
        gmm_iv_loss = (Pi @ mse).mean()
        return gmm_iv_loss
