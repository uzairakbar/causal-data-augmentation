import os
import numpy as np
from typing import Dict, Tuple
from numpy.typing import NDArray

from src.sem.abstract import StructuralEquationModel as SEM
from src.regressors.erm import LeastSquaresClosedForm as ERM


class OpticalDeviceSEM(SEM):
    @staticmethod
    def load_dataset(directory: str='data/linear') -> Dict[int, NDArray]:
        file_list = os.listdir(directory)
        file_list = [f for f in file_list if 'confounder' in f and 'random' not in f]
        
        dataset = {}
        for experiment, file_name in enumerate(file_list):
            dataset[experiment] = np.genfromtxt(
                f'{directory}/{file_name}', delimiter=' '
            )
        return dataset
    
    _DATASET: Dict[int, NDArray] = load_dataset.__func__()

    def __init__(
            self, experiment: int=0, center: bool=True
        ):
        experiment_data = self.get_experiment_data(experiment)

        if center:
            experiment_data -= experiment_data.mean(axis = 0)

        y = experiment_data[:, -1:]
        XH = experiment_data[:, :-1]

        W_XHY = ERM().fit(XH, y).solution

        self.W_XY = W_XHY[:-1, :]
        self.y, self.X = y, XH[:, :-1]
    
    def sample(self, N: int=1, **kwargs) -> Tuple[NDArray, NDArray]:
        N_max, M = self.X.shape
        indices = np.arange(N_max)
        replace = N > N_max
        sampled = np.random.choice(indices,
                                   N,
                                   replace)
        return self.X[sampled], self.y[sampled]
    
    @classmethod
    def get_experiment_data(cls, n: int) -> NDArray:
        return cls._DATASET[n]
    
    @classmethod
    def num_experiments(cls) -> int:
        return len(cls._DATASET)
