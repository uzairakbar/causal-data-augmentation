import os
import numpy as np

from src.sem.abstract import StructuredEquationModel as SEM
from src.regressors.erm import LeastSquaresClosedForm as ERM


class OpticalDeviceSEM(SEM):
    @staticmethod
    def load_dataset(directory="data/linear"):
        file_list = os.listdir(directory)
        file_list = [f for f in file_list if "confounder" in f and "random" not in f]
        
        dataset = {}
        for experiment, file_name in enumerate(file_list):
            dataset[experiment] = np.genfromtxt(f"{directory}/{file_name}",
                                                delimiter=" ")
        return dataset
    
    _DATASET = load_dataset.__func__()

    def __init__(self,
                 experiment=0,
                 center = True):
        experiment_data = self.get_experiment_data(experiment)

        if center:
            experiment_data -= experiment_data.mean(axis = 0)

        y = experiment_data[:, -1:]
        XH = experiment_data[:, :-1]

        W_XHY = ERM().fit(XH, y).solution

        self.W_XY = W_XHY[:-1, :]
        self.y, self.X = y, XH[:, :-1]

        return super(OpticalDeviceSEM, self).__init__()
    
    def sample(self, N = 1, **kwargs):
        N_max, M = self.X.shape
        indices = np.arange(N_max)
        replace = N > N_max
        sampled = np.random.choice(indices,
                                   N,
                                   replace)
        return self.X[sampled], self.y[sampled]
    
    @classmethod
    def get_experiment_data(cls, n):
        return cls._DATASET[n]
    
    @classmethod
    def num_experiments(cls):
        return len(cls._DATASET)