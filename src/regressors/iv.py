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

