import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data_utils

from src.regressors.abstract import EmpiricalRiskMinimizer as ERM


class LeastSquaresClosedForm(ERM):
    def fit(self, X, y):
        self._W = np.linalg.pinv(X) @ y
        return self


class LeastSquaresGradientDescent(ERM):
    _models = {
        "linear": lambda input_dim: torch.nn.Linear(input_dim, 1, bias=False),
        "2-layer": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(20, 1)
        ),
    }

    def __init__(self,
                 model="linear",
                 epochs=200):
        super().__init__()

        if model in self._models:
            self.__model = self._models[model]
        else:
            raise ValueError("model has invalid value " + str(model))
        self._optimizer = None
        self._epochs = epochs

    def fit_f_minibatch(self, train):
        losses = list()
        for i, (X_b, y_b) in enumerate(train):
            if torch.cuda.is_available():
                X_b = X_b.cuda()
                y_b = y_b.cuda()
            loss_val = self._optimizer.step(lambda: self.loss(X_b, y_b))
            losses += [loss_val.data.cpu().numpy()]
        print("  train loss ", np.mean(losses))

    def fit_f_batch(self, X, y):
        _ = self._optimizer.step(lambda: self.loss(X, y))

    def fit(self, X, y):
        n, m = X.shape

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        self.f = self.__model(m)
        if torch.cuda.is_available():
            self.f = self.f.cuda()
        self.f.float()
        self._optimizer = torch.optim.Adam(self.f.parameters(), lr=0.01)

        batch_mode = "mini" if n >= 1000 else "full"
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y),
                                      batch_size=128, shuffle=True)

        for epoch in range(self._epochs):
            if batch_mode == "full":
                self.fit_f_batch(X, y)
            else:
                print("g epoch %d / %d" % (epoch + 1, self._epochs))
                self.fit_f_minibatch(train)
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        if torch.cuda.is_available():
            X = X.cuda()
        return self.f(X).data.cpu().numpy()
    
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
    def _loss(X, y, f):
        y_hat = f(X)
        loss = F.mse_loss(y, y_hat, reduction='sum')
        return loss

