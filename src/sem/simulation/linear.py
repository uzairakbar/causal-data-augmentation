import numpy as np

from src.sem.abstract import StructuredEquationModel as SEM


# specify default parameters
COVARIATE_DIMENSION = 30
CONFOUNDER_DIMENSION = COVARIATE_DIMENSION
LABEL_DIMENSION = 1


class LinearSimulationSEM(SEM):
    def __init__(self,
                 covariate_dimension = COVARIATE_DIMENSION,
                 confounder_dimension = CONFOUNDER_DIMENSION,
                 label_dimension = LABEL_DIMENSION):
        self.covariate_dimension = covariate_dimension
        self.confounder_dimension = confounder_dimension
        self.label_dimension = label_dimension

        self.W_CX = np.random.randn(confounder_dimension, covariate_dimension)
        self.W_CY = np.random.randn(confounder_dimension, label_dimension)
        self.W_XY = np.random.randn(covariate_dimension, label_dimension)
        
        super(LinearSimulationSEM, self).__init__()
    
    def sample(self, N = 1, lamda = 1.0, intervention = False, **kwargs):
        C = np.random.randn(N, self.confounder_dimension)

        if intervention:
            X = np.random.randn(N, self.covariate_dimension)
        else:
            N_X = np.random.randn(N, self.covariate_dimension)*0.1
            X = C @ self.W_CX + N_X

        N_Y = np.random.randn(N, self.label_dimension)*0.1
        Y = X @ self.W_XY + lamda * C @ self.W_CY + N_Y
        return X, Y

