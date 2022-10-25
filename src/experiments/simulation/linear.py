import numpy as np
from abc import ABC, abstractmethod

from src.data_augmentors.simulation.linear import NullSpaceTranslation as DA

from src.sem.simulation.linear import LinearSimulationSEM as SEM

from src.regressors.daiv import DAIVLeastSquaresClosedForm as DAIV
from src.regressors.erm import LeastSquaresClosedForm as ERM
from src.regressors.iv import IVTwoStageLeastSquares as IV
from src.regressors.daiv import DAIVProjectedLeastSquares as DAIVP

from src.regressors.model_selectors import LeaveOneOut as LOO
from src.regressors.model_selectors import ConfounderCorrection as CC

from src.experiments.utils import (
    set_seed,
    sweep_plot,
    relative_sq_error,
)


ALL_METHODS = {
    "ERM": lambda: ERM(),
    "DA+ERM": lambda: ERM(),
    "DAIV+LOO": lambda: LOO(
        estimator=DAIV(),
        param_distributions = {"alpha": np.random.lognormal(1, 1, 10)},
        cv=5                                # TODO: proper LOO CV
    ),
    "DAIV+CC": lambda: CC(estimator=DAIV()),
    "DA+IV": lambda: IV(),
}


class Experiment(ABC):
    def __init__(self,
                 x_dimension = 30,
                 n_samples = 500,
                 n_experiments = 10,
                 seed = 42,
                 methods = "all",
                 sweep_samples = 10):
        self.x_dimension = x_dimension
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.seed = seed
        self.sweep_samples = sweep_samples
        if methods == "all":
            self.methods = ALL_METHODS
        else:
            self.methods = {m: ALL_METHODS[m] for m in methods.split(',')}
    
    @staticmethod
    def fit(method_name, method, X, y, G, GX, param):
        if method_name == "DAIV":
            model = method(alpha = param)
        else:
            model = method()
        
        if "ERM" in method_name:
            if "DA" in method_name:
                model.fit(X=GX, y=y)
            else:
                model.fit(X=X, y=y)
        elif "DAIV" in method_name:
            model.fit(X=X, y=y, G=G, GX=GX)
        else:
            model.fit(X=GX, y=y, Z=G)
        
        return model
    
    @abstractmethod
    def generate_dataset(self, sem, da, param):
        pass

    @abstractmethod
    def param_sweep(self):
        pass

    def compute_result(self,
               sem_solution,
               method_name,
               method,
               X, y, G, GX,
               param):
        model = self.fit(
            method_name, method, X, y, G, GX, param
        )
        error = relative_sq_error(sem_solution, model.solution)
        return error

    def run_experiment(self):
        if self.seed >= 0:
            set_seed(self.seed)
        param_values = self.param_sweep()

        all_sems = []
        all_augmenters = []
        for _ in range(self.n_experiments):
            sem = SEM(self.x_dimension)
            da = DA(sem.W_XY)
            all_sems.append(sem)
            all_augmenters.append(da)
        
        error_dim = (self.sweep_samples, self.n_experiments)
        results = {name: np.zeros(error_dim) for name in self.methods}

        for i, param in enumerate(param_values):
            for j, (sem, da) in enumerate(zip(all_sems, all_augmenters)):
                sem_solution = sem.solution

                X, y, G, GX = self.generate_dataset(sem, da, param)
                for method_name, method in self.methods.items():
                    results[method_name][i][j] = self.compute_result(
                        sem_solution, method_name, method, X, y, G, GX, param
                    )
        return param_values, results


class LambdaSweep(Experiment):
    def generate_dataset(self, sem, da, param):
        X, y = sem(N = self.n_samples, lamda = param)
        GX, G = da(X)
        return X, y, G, GX

    def param_sweep(self):
        lambda_values = np.linspace(
            0, 1, num=self.sweep_samples
        )
        return lambda_values


class GammaSweep(Experiment):
    def generate_dataset(self, sem, da, param):
        X, y = sem(N = self.n_samples)
        GX, G = da(X, gamma = param)
        return X, y, G, GX

    def param_sweep(self):
        gamma_values = np.logspace(
            -2, 2, base=10, num=self.sweep_samples
        )
        return gamma_values


class AlphaSweep(Experiment):
    def __init__(self,
                 **kwargs):
        super(AlphaSweep, self).__init__(**kwargs)
        self.methods["DAIV"] = (
            lambda alpha: DAIV(alpha = alpha)
        )

    def generate_dataset(self, sem, da, param):
        X, y = sem(N = self.n_samples)
        GX, G = da(X)
        return X, y, G, GX

    def param_sweep(self):
        alpha_values = np.logspace(
            -5, 5, base=10, num=self.sweep_samples
        )
        return alpha_values
    
    def compute_result(self,
               sem_solution,
               method_name,
               method,
               X, y, G, GX,
               param):
        model = self.fit(
            method_name, method, X, y, G, GX, param
        )
        error = relative_sq_error(sem_solution, model.solution)
        if "DAIV+" in method_name:
            return model.alpha
        else:
            return error


def main():
    lambda_values, results = LambdaSweep().run_experiment()
    sweep_plot(
        lambda_values, results, xlabel=r"$\lambda$", xscale="linear"
    )

    gamma_values, results = GammaSweep().run_experiment()
    sweep_plot(
        gamma_values, results, xlabel=r"$\gamma$", xscale="log"
    )

    alpha_values, results = AlphaSweep().run_experiment()
    vertical_plots = ([
        method for method in ALL_METHODS.keys() if "DAIV+" in method
    ])
    sweep_plot(
        alpha_values, results, xlabel=r"$\alpha$", xscale="log",
        vertical_plots=vertical_plots, trivial_solution=False
    )


if __name__ == "__main__":
    main()

