import torch
import numpy as np
import torch.utils.data as data_utils

from src.regressors.abstract import IVRegressor as IV
from src.regressors.erm import (
    LeastSquaresClosedForm as LSCF,
    LeastSquaresGradientDescent as LSGD,
)


ERM = {"cf": LSCF(),
       "gd": LSGD()}


class IVTwoStageLeastSquares(IV):
    def __init__(self, s1 = "cf", s2 = "gd"):
        self.s1 = s1
        self.s2 = s2
        super(IVTwoStageLeastSquares, self).__init__()
    
    def fit(self, X, y, Z = None):

        S1 = ERM[self.s1].fit(Z, X).solution
        Xhat = Z @ S1

        S2 = ERM[self.s2].fit(Xhat, y).solution
        self._W = S2

        return self


class IVGeneralizedMomentMethod(IV):
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
                 gmm_steps=20,
                 epochs=200):
        if model in self._models:
            self.__model = self._models[model]
        else:
            raise ValueError("model has invalid value " + str(model))
        self._optimizer = None
        self._gmm_steps = gmm_steps
        self._epochs = epochs
        super(IVGeneralizedMomentMethod, self).__init__()

    def fit_f_minibatch(self, train, weight):
        losses = list()
        for i, (X_b, y_b, Z_b) in enumerate(train):
            if torch.cuda.is_available():
                X_b = X_b.cuda()
                y_b = y_b.cuda()
                Z_b = Z_b.cuda()
                weights = weights.cuda()
            loss_val = self._optimizer.step(lambda: self.loss(X_b, y_b, Z_b, weight))
            losses += [loss_val.data.cpu().numpy()]
        print("  train loss ", np.mean(losses))

    def fit_f_batch(self, X, y, Z, weight):
        _ = self._optimizer.step(lambda: self.loss(X, y, Z, weight))

    def fit(self, X, y, Z):
        n, m = X.shape
        _, k = Z.shape

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        Z = torch.tensor(Z, dtype=torch.float)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
            Z = Z.cuda()
        
        self.f = self.__model(m)
        if torch.cuda.is_available():
            self.f = self.f.cuda()
        self.f.float()
        self._optimizer = torch.optim.Adam(self.f.parameters(), lr=0.01)
        weights = torch.eye(k)
        if torch.cuda.is_available():
            weights = weights.cuda()

        batch_mode = "mini" if n >= 1000 else "full"
        train = data_utils.DataLoader(data_utils.TensorDataset(X, y, Z),
                                      batch_size=128, shuffle=True)

        for step in range(self._gmm_steps):
            print("GMM step %d/%d" % (step + 1, self._gmm_steps))
            if step > 0:
                # optimize weights
                with torch.no_grad():
                    moment_conditions = Z.mul(y - self.f(X))
                    covariance_matrix = torch.mm(moment_conditions.t(),
                                                 moment_conditions) / n
                    weights = torch.as_tensor(
                        np.linalg.pinv(covariance_matrix.cpu().numpy(),
                                       rcond=1e-9))

            for epoch in range(self._epochs):
                if batch_mode == "full":
                    self.fit_f_batch(X, y, Z, weights)
                else:
                    print("g epoch %d / %d" % (epoch + 1, self._epochs))
                    self.fit_f_minibatch(train, weights)
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
             X, y, Z,
             weights):
        loss = self._loss(X, y, Z, self.f, weights)
        self._optimizer.zero_grad()
        loss.backward()
        return loss

    @staticmethod
    def _loss(X, y, Z, f,
              weights):
        moment_conditions = Z.mul(y - f(X))
        moms = moment_conditions.mean(dim=0, keepdim=True)
        loss = torch.mm(torch.mm(moms, weights), moms.t())
        return loss

