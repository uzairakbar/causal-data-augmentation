import torch
import numpy as np
import cvxpy as cp
from loguru import logger
import torch.nn.functional as F
import torch.utils.data as data_utils

from src.regressors.utils import MODELS, Model, device

from src.regressors.abstract import EmpiricalRiskMinimizer as ERM


DEVICE: str=device()
MAX_BATCH: int=1_000


class LeastSquaresClosedForm(ERM):
    def _fit(self, X, y, **kwargs):
        self._W = np.linalg.pinv(X) @ y
        return self
    
    def _predict(self, X):
        return X @ self._W


class LeastSquaresCvxpy(ERM):
    def _fit(self, X, y):
        h0 = np.linalg.pinv(X) @ y
        h = cp.Variable(h0.shape)
        cost = cp.norm(y - X @ h)
        prob = cp.Problem(
            cp.Minimize(cost)
        )
        result = prob.solve(solver=cp.CLARABEL)
        self._W = h.value
        return self
    
    def _predict(self, X):
        return X @ self._W


class LeastSquaresGradientDescent(ERM):
    _models = MODELS

    def __init__(self, model: Model='linear'):
        super().__init__()

        if model in self._models:
            self.__model = self._models[model]
        else:
            raise ValueError(f'model has invalid value {model}')
        self._optimizer = None

    def fit_f_minibatch(self, train):
        losses = list()
        for i, (X_b, y_b) in enumerate(train):
            X_b, y_b = (
                X_b.to(DEVICE), y_b.to(DEVICE)
            )
            loss_val = self._optimizer.step(lambda: self.loss(X_b, y_b))
            losses += [loss_val.data.cpu().numpy()]
        logger.info(f'  train loss {np.mean(losses):.2f}')

    def fit_f_batch(self, X, y):
        _ = self._optimizer.step(lambda: self.loss(X, y))

    def _fit(
            self,
            X, y,
            lr=0.001, batch=512, epochs=40,
            pbar_manager=None
        ):
        n, m = X.shape

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

        X, y = (
            X.to(DEVICE), y.to(DEVICE)
        )

        batch_mode = 'mini' if n > MAX_BATCH else 'full'
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y),
                                      batch_size=batch, shuffle=True)
        
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
        X = X.to(DEVICE)
        
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
             X, y):
        loss = self._loss(X, y, self.f)
        self._optimizer.zero_grad()
        loss.backward()
        return loss

    @staticmethod
    def _loss(X, y, f, reduction='sum'):
        y_hat = f(X)
        if isinstance(f[-1], torch.nn.LogSoftmax):
            loss = F.nll_loss(y_hat, y.flatten(), reduction=reduction)
        elif isinstance(f[-1], torch.nn.Sigmoid):
            loss = F.binary_cross_entropy(y_hat, y, reduction=reduction)
        else:
            loss = F.mse_loss(y_hat, y, reduction=reduction)
        return loss

