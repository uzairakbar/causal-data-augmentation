import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data_utils

from src.regressors.abstract import EmpiricalRiskMinimizer as ERM


class LeastSquaresClosedForm(ERM):
    def _fit(self, X, y):
        self._W = np.linalg.pinv(X) @ y
        return self
    
    def _predict(self, X):
        return X @ self._W


class LeastSquaresGradientDescent(ERM):
    _models = {
        "linear": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1, bias=False)
        ),
        "2-layer": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(20, 1)
        ),
        "cmnist": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        ),
        "rmnist": lambda input_dim: torch.nn.Sequential(
            torch.nn.Unflatten(1, torch.Size([1, 28, 28])),
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False),
            torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False),
            torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.LogSoftmax(dim=1)
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
            elif torch.backends.mps.is_available():
                X_b = X_b.to("mps")
                y_b = y_b.to("mps")
            
            loss_val = self._optimizer.step(lambda: self.loss(X_b, y_b))
            losses += [loss_val.data.cpu().numpy()]
        print("  train loss ", np.mean(losses))

    def fit_f_batch(self, X, y):
        _ = self._optimizer.step(lambda: self.loss(X, y))

    def _fit(self, X, y):
        n, m = X.shape

        self.f = self.__model(m)
        self.f.float()
        self.f.train()
        if torch.cuda.is_available():
            self.f = self.f.cuda()
        elif torch.backends.mps.is_available():
            self.f = self.f.to("mps")

        self._optimizer = torch.optim.Adam(self.f.parameters(), lr=0.01)

        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.float)
        X = torch.tensor(X, dtype=torch.float)
        
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            X = X.to("mps")
            y = y.to("mps")
            print("Using MPS")
        else:
            print("Using CPU")

        batch_mode = "mini" if n >= 1000 else "full"
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y),
                                      batch_size=128, shuffle=True)

        for epoch in range(self._epochs):
            if batch_mode == "full":
                self.fit_f_batch(X, y)
            else:
                print("g epoch %d / %d" % (epoch + 1, self._epochs))
                self.fit_f_minibatch(train)
        self.f.eval()
        return self

    def _predict(self, X):
        X = torch.tensor(X, dtype=torch.float)
        
        if torch.cuda.is_available():
            X = X.cuda()
        elif torch.backends.mps.is_available():
            X = X.to("mps")
        
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
    def _loss(X, y, f, reduction="sum"):
        y_hat = f(X)
        if isinstance(f[-1], torch.nn.LogSoftmax):
            loss = F.nll_loss(y_hat, y.flatten(), reduction=reduction)
        elif isinstance(f[-1], torch.nn.Sigmoid):
            loss = F.binary_cross_entropy(y_hat, y, reduction=reduction)
        else:
            loss = F.mse_loss(y_hat, y, reduction=reduction)
        return loss

