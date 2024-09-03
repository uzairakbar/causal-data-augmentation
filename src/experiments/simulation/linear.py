import argparse
import enlighten
import numpy as np
from loguru import logger
from enlighten import Manager
from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional

from src.data_augmentors.simulation.linear import NullSpaceTranslation as DA

from src.sem.simulation.linear import LinearSimulationSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector
from src.regressors.daiv import DAIVLeastSquaresClosedForm as DAIValpha
from src.regressors.erm import LeastSquaresClosedForm as ERM
from src.regressors.iv import IVTwoStageLeastSquares as IV
from src.regressors.daiv import DAIVProjectedLeastSquares as DAIVPi
from src.regressors.daiv import DAIVConstrainedLeastSquares as DAIV

from src.regressors.model_selectors import LeaveOneOut as LOO
from src.regressors.model_selectors import ConfounderCorrection as CC

from src.experiments.utils import (
    save,
    set_seed,
    bootstrap,
    sweep_plot,
    relative_sq_error,
)


ALL_METHODS: Dict[str, Callable[[Optional[float]], Regressor | ModelSelector]] = {
    'ERM': lambda: ERM(),
    'DA+ERM': lambda: ERM(),
    'DAIV+LOO': lambda: LOO(
        estimator=DAIValpha(),
        param_distributions = {'alpha': np.random.lognormal(1, 1, 10)},
        cv=5,                                # TODO: proper LOO CV
        n_jobs=-1,
    ),
    'DAIV+CC': lambda: CC(estimator=DAIValpha()),
    'DAIVPi': lambda: DAIVPi(),
    'DAIV': lambda: DAIV(),
    'DA+IV': lambda: IV(),
}
MANAGER = enlighten.get_manager()


class Experiment(ABC):
    def __init__(
            self,
            seed: int,
            n_samples: int,
            x_dimension: int,
            n_experiments: int,
            sweep_samples: int,
            methods: Dict[str, Callable[[Optional[float]], Regressor | ModelSelector]]
        ):
        self.x_dimension = x_dimension
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.seed = seed
        self.sweep_samples = sweep_samples
        self.methods = methods
    
    @staticmethod
    def fit(method_name: str,
            method: Callable[[Optional[str]], Regressor | ModelSelector],
            X, y, G, GX,
            param: float) -> Regressor | ModelSelector:
        if method_name == 'DAIValpha':
            model = method(alpha = param)
        else:
            model = method()
        
        if 'ERM' in method_name:
            if 'DA' in method_name:
                model.fit(X=GX, y=y)
            else:
                model.fit(X=X, y=y)
        elif 'DAIV' in method_name:
            model.fit(X=X, y=y, G=G, GX=GX)
        else:
            model.fit(X=GX, y=y, Z=G)
        
        return model
    
    @abstractmethod
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        pass

    @abstractmethod
    def param_sweep(self):
        pass

    def compute_result(self,
               sem_solution,
               method_name: str,
               method: Callable[[Optional[str]], Regressor | ModelSelector],
               X, y, G, GX,
               param: float) -> float:
        model = self.fit(
            method_name, method, X, y, G, GX, param
        )
        error = relative_sq_error(sem_solution, model.solution)**0.5
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
        
        experiment_name = self.__class__.__name__
        pbar_experiment = manager.counter(
            total=self.sweep_samples, desc=f'{experiment_name}', unit='params'
        )
        for i, param in enumerate(param_values):

            pbar_sem = manager.counter(
                total=self.n_experiments, desc=f'Param. {param:.2f}', unit='experiments', leave=False
            )
            for j, (sem, da) in enumerate(zip(all_sems, all_augmenters)):
                sem_solution = sem.solution

                X, y, G, GX = self.generate_dataset(sem, da, param)
                
                pbar_methods = manager.counter(
                    total=len(self.methods), desc=f'SEM {j}', unit='methods', leave=False
                )
                for method_name, method in self.methods.items():
                    results[method_name][i][j] = self.compute_result(
                        sem_solution, method_name, method, X, y, G, GX, param
                    )

                    pbar_methods.update()
                pbar_methods.close()
                pbar_sem.update()
            pbar_sem.close()
            pbar_experiment.update()
        pbar_experiment.close()
        return param_values, results


class LambdaSweep(Experiment):
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        X, y = sem(N = self.n_samples, lamda = param)
        GX, G = da(X)
        return X, y, G, GX

    def param_sweep(self):
        lambda_values = np.linspace(
            0, 1, num=self.sweep_samples
        )
        return lambda_values


class GammaSweep(Experiment):
    def generate_dataset(self, sem: SEM, da: DA, param: float):
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
        self.methods['DAIValpha'] = (
            lambda alpha: DAIValpha(alpha = alpha)
        )

    def generate_dataset(self, sem: SEM, da: DA, param: float):
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
               method_name: str,
               method: Callable[[Optional[str]], Regressor | ModelSelector],
               X, y, G, GX,
               param: float) -> float:
        model = self.fit(
            method_name, method, X, y, G, GX, param
        )
        error = relative_sq_error(sem_solution, model.solution)**0.5
        if 'DAIV+' in method_name:
            return model.alpha
        else:
            return error


def run(
        seed: int,
        n_samples: int,
        x_dimension: int,
        n_experiments: int,
        sweep_samples: int,
        methods: str,
        manager: Manager=MANAGER
    ):
    status = manager.status_bar(
        status_format=u'Linear simulation{fill}Sweeping {sweep}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, sweep='<parameter>',
        autorefresh=True, min_delta=0.5
    )
    
    if methods == 'all':
        methods = ALL_METHODS
    else:
        methods = {m: ALL_METHODS[m] for m in methods.split(',')}
    
    # sweep over lambda parameter
    status.update(sweep='lambda')
    logger.info('Sweeping over lambda parameters.')
    lambda_values, results = LambdaSweep(
        seed=seed,
        x_dimension=x_dimension,
        n_samples=n_samples,
        n_experiments=n_experiments,
        methods=methods,
        sweep_samples=sweep_samples
    ).run_experiment()
    sweep_plot(
        lambda_values, bootstrap(results), xlabel=r'$\lambda$', xscale='linear'
    )
    save(lambda_values, 'lambda_values')
    save(results, 'lambda_results')

    # sweep over gamma parameter
    status.update(sweep='gamma')
    logger.info('Sweeping over gamma parameters.')
    gamma_values, results = GammaSweep(
        seed=seed,
        x_dimension=x_dimension,
        n_samples=n_samples,
        n_experiments=n_experiments,
        methods=methods,
        sweep_samples=sweep_samples
    ).run_experiment()
    sweep_plot(
        gamma_values, bootstrap(results), xlabel=r'$\gamma$', xscale='log'
    )
    save(gamma_values, 'gamma_values')
    save(results, 'gamma_results')

    # sweep over alpha parameter
    status.update(sweep='alpha')
    logger.info('Sweeping over alpha parameters.')
    alpha_values, results = AlphaSweep(
        seed=seed,
        x_dimension=x_dimension,
        n_samples=n_samples,
        n_experiments=n_experiments,
        methods=methods,
        sweep_samples=sweep_samples
    ).run_experiment()
    vertical_plots = ([
        method for method in ALL_METHODS.keys() if 'DAIV+' in method
    ])
    sweep_plot(
        alpha_values, bootstrap(results), xlabel=r'$\alpha$', xscale='log',
        vertical_plots=vertical_plots, trivial_solution=False
    )
    save(alpha_values, 'alpha_values')
    save(results, 'alpha_results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear simulation experiment.')
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    parser.add_argument(
        '--n_samples', type=int, default=2000, help='Number of samples per experiment.'
    )
    parser.add_argument(
        '--x_dimension', type=int, default=30, help='Dimension of treatment.'
    )
    parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments.')
    parser.add_argument(
        '--sweep_samples', type=int, default=10, help='Sweep resolution across lambda, alpha and gamma.'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default='all',
        help='Methods to use. Specify in comma-separated format -- "ERM,DA+ERM,DA+UIV,DA+IV". Default is "all".'
    )
    args = parser.parse_args()
    run(**vars(args))
