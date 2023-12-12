import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data_utils

from src.regressors.abstract import IVRegressor as IV
from src.regressors.erm import (
    LeastSquaresClosedForm as LSCF,
    LeastSquaresGradientDescent as LSGD,
    # LeastSquaresCvxpy as LSGD,
)


ERM = {"cf": LSCF(),
       "gd": LSGD()}


class IVTwoStageLeastSquares(IV):
    def __init__(self, s1 = "cf", s2 = "gd"):
        self.s1 = s1
        self.s2 = s2
        super(IVTwoStageLeastSquares, self).__init__()
    
    def _fit(self, X, y, Z = None):

        S1 = ERM[self.s1].fit(Z, X).solution
        Xhat = Z @ S1

        S2 = ERM[self.s2].fit(Xhat, y).solution
        self._W = S2

        return self
    
    def _predict(self, X):
        return X @ self._W


class IVGeneralizedMomentMethod(IV):
    _models = {
        "linear": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1, bias=False)
        ),
        "2-layer": lambda input_dim: torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(20, 1)
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
                 gmm_steps=10,
                 epochs=100):
        if model in self._models:
            self.__model = self._models[model]
        else:
            raise ValueError("model has invalid value " + str(model))
        self._optimizer = None
        self._gmm_steps = gmm_steps
        self._epochs = epochs
        super(IVGeneralizedMomentMethod, self).__init__()

    def fit_f_minibatch(self, train, weights):
        losses = list()
        for i, (X_b, y_b, Z_b) in enumerate(train):
            if torch.cuda.is_available():
                X_b = X_b.cuda()
                y_b = y_b.cuda()
                Z_b = Z_b.cuda()
                weights = weights.cuda()
            elif torch.backends.mps.is_available():
                X_b = X_b.to("mps")
                y_b = y_b.to("mps")
                Z_b = Z_b.to("mps")
                weights = weights.to("mps")
            loss_val = self._optimizer.step(
                lambda: self.loss(X_b, y_b, Z_b, weights)
            )
            losses += [loss_val.data.cpu().numpy()]
        print("  train loss ", np.mean(losses))

    def fit_f_batch(self, X, y, Z, weights):
        _ = self._optimizer.step(lambda: self.loss(X, y, Z, weights))

    def _fit(self, X, y, Z):
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
        if torch.cuda.is_available():
            self.f = self.f.cuda()
        elif torch.backends.mps.is_available():
            self.f = self.f.to("mps")
        
        self._optimizer = torch.optim.Adam(self.f.parameters(), lr=0.001)
        
        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.float)
        X = torch.tensor(X, dtype=torch.float)
        Z = torch.tensor(Z, dtype=torch.float)

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
            Z = Z.cuda()
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            X = X.to("mps")
            y = y.to("mps")
            Z = Z.to("mps")
            print("Using MPS")
        else:
            print("Using CPU")
        
        if isinstance(self.f[-1], torch.nn.LogSoftmax):
            weights = torch.eye(10*k)
        else:
            weights = torch.eye(k)
        if torch.cuda.is_available():
            weights = weights.cuda()
        elif torch.backends.mps.is_available():
            weights = weights.to("mps")

        batch_mode = "mini" if n >= 1000 else "full"
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y, Z),
                                      batch_size=128, shuffle=True)

        for step in range(self._gmm_steps):
            print("GMM step %d/%d" % (step + 1, self._gmm_steps))
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
                    #         + 1e-7*torch.eye(covariance_matrix.size(dim=-1), device="mps")
                    #     )
                    # )
                    weights = torch.as_tensor(
                        np.linalg.pinv(
                            covariance_matrix.cpu().numpy() + 1e-7*np.eye(covariance_matrix.size(dim=-1)),
                            rcond=1e-9
                        )
                    )
                    if torch.cuda.is_available():
                        weights = weights.cuda()
                    elif torch.backends.mps.is_available():
                        weights = weights.to("mps")

            for epoch in range(self._epochs):
                if batch_mode == "full":
                    self.fit_f_batch(X, y, Z, weights)
                else:
                    print("g epoch %d / %d" % (epoch + 1, self._epochs))
                    self.fit_f_minibatch(train, weights)

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

