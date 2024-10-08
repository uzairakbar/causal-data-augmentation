import copy
import enlighten
import numpy as np
from loguru import logger
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List

from src.data_augmentors.simulation.linear import NullSpaceTranslation as DA

from src.sem.simulation.linear import LinearSimulationSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector

from src.regressors.erm import LeastSquaresClosedForm as ERM

from src.regressors.iv import TwoStageLeastSquaresIV as IV

from src.regressors.uiv import (
    ConstrainedLeastSquaresUnfaithfulIV as UIV,
    LeastSquaresClosedFormUnfaithfulIV as UIV_a,
    ProjectedLeastSquaresUnfaithfulIV as UIV_Pi,
)

from src.regressors.baselines import (
    RICE,
    MiniMaxREx as MMREx,
    VarianceREx as VREx,
    LinearAnchorRegression as AR,
    InvariantRiskMinimization as IRM,
    InvariantCausalPrediction as ICP,
    DistributionallyRobustOptimization as DRO,
)

from src.regressors.model_selectors import (
    LevelCV,
    VanillaCV as CV,
    ConfounderCorrection as CC
)

from src.experiments.utils import (
    save,
    set_seed,
    box_plot,
    tex_table,
    fit_model,
    sweep_plot,
    relative_error,
    ANNOTATE_BOX_PLOT,
    ANNOTATE_SWEEP_PLOT,
)


ModelBuilder = Callable[[Optional[float]], Regressor | ModelSelector]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='linear_simulation'
DEFAULT_CV_SAMPLES: int=10
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_FOLDS: int=5
DEFAULT_CV_JOBS: int=1


class SweepExperiment(ABC):
    def __init__(
            self,
            seed: int,
            n_samples: int,
            kernel_dim: int,
            n_experiments: int,
            sweep_samples: int,
            methods: Dict[str, Callable[[Optional[float]], Regressor | ModelSelector]]
        ):
        self.seed = seed
        self.n_samples = n_samples
        self.kernel_dim = kernel_dim
        self.n_experiments = n_experiments
        self.sweep_samples = sweep_samples
        self.methods = methods
    
    @staticmethod
    def fit(method_name: str,
            method: Callable[[Optional[str]], Regressor | ModelSelector],
            X, y, G, GX,
            param: float,
            da: Optional[DA]=None) -> Regressor | ModelSelector:
        if method_name == 'DA+UIV-a':
            model = method(alpha = param)
        else:
            model = method()
        
        fit_model(
            model=model,
            name=method_name,
            X=X, y=y, G=G, GX=GX, da=da
        )
        
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
               param: float,
               da: Optional[DA]=None) -> float:
        model = self.fit(
            method_name, method, X, y, G, GX, param, da=da
        )
        error = relative_error(sem_solution, model.solution)
        return error

    def run_experiment(self):
        if self.seed >= 0:
            set_seed(self.seed)
        param_values = self.param_sweep()

        all_sems = []
        all_augmenters = []
        for _ in range(self.n_experiments):
            sem = SEM()
            da = DA(sem.W_XY, kernel_dim=self.kernel_dim)
            all_sems.append(sem)
            all_augmenters.append(da)
        
        error_dim = (self.sweep_samples, self.n_experiments)
        results = {name: np.zeros(error_dim) for name in self.methods}
        
        experiment_name = self.__class__.__name__
        pbar_experiment = MANAGER.counter(
            total=self.sweep_samples, desc=f'{experiment_name}', unit='params'
        )
        for i, param in enumerate(param_values):

            pbar_sem = MANAGER.counter(
                total=self.n_experiments, desc=f'Param. {param:.2f}', unit='experiments', leave=False
            )
            for j, (sem, da) in enumerate(zip(all_sems, all_augmenters)):
                sem_solution = sem.solution

                X, y, G, GX = self.generate_dataset(sem, da, param)
                
                pbar_methods = MANAGER.counter(
                    total=len(self.methods), desc=f'SEM {j}', unit='methods', leave=False
                )
                for method_name, method in self.methods.items():
                    results[method_name][i][j] = self.compute_result(
                        sem_solution, method_name, method, X, y, G, GX, param, da=da
                    )

                    pbar_methods.update()
                pbar_methods.close()
                pbar_sem.update()
            pbar_sem.close()
            pbar_experiment.update()
        pbar_experiment.close()
        return param_values, results


class LambdaSweep(SweepExperiment):
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        X, y = sem(N = self.n_samples, lamda = param)
        GX, G = da(X)
        return X, y, G, GX

    def param_sweep(self):
        lambda_values = np.linspace(
            0, 1, num=self.sweep_samples
        )
        return lambda_values


class GammaSweep(SweepExperiment):
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        X, y = sem(N = self.n_samples)
        GX, G = da(X, gamma = param)
        return X, y, G, GX

    def param_sweep(self):
        gamma_values = np.logspace(
            -2.5, 1.5, base=10, num=self.sweep_samples
        )
        return gamma_values


class AlphaSweep(SweepExperiment):
    def __init__(self, **kwargs):
        super(AlphaSweep, self).__init__(**kwargs)
        self.methods = {
            **copy.deepcopy(self.methods),
            'DA+UIV-a': lambda alpha: UIV_a(alpha = alpha)
        }

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
               param: float,
               da: Optional[DA]=None) -> float:
        model = self.fit(
            method_name, method, X, y, G, GX, param
        )
        error = relative_error(sem_solution, model.solution)
        if ('DA+UIV-' in method_name) and (method_name != 'DA+UIV-a'):
            return model.alpha
        else:
            return error


class BaselineExperiment(SweepExperiment):
    def __init__(self, **kwargs):
        super(BaselineExperiment, self).__init__(
            sweep_samples=1, **kwargs
        )

    def generate_dataset(self, sem: SEM, da: DA, param: float):
        X, y = sem(N = self.n_samples)
        GX, G = da(X, gamma = param)
        return X, y, G, GX

    def param_sweep(self):
        gamma = [1.0]
        return gamma


def run(
        seed: int,
        n_samples: int,
        kernel_dim: int,
        n_experiments: int,
        sweep_samples: int,
        methods: List[str],
        augmentations: Optional[List[str]]=None,
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
    ):
    status = MANAGER.status_bar(
        status_format=u'Linear simulation{fill}Sweeping {sweep}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, sweep='<parameter>',
        autorefresh=True, min_delta=0.5
    )

    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ERM': lambda: ERM(),
        'DA+ERM': lambda: ERM(),
        'DA+UIV-CV': lambda: CV(
            estimator=UIV_a(),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DA+UIV-LCV': lambda: LevelCV(
            metric='mse',
            estimator=UIV_a(),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DA+UIV-CC': lambda: CC(
            estimator=UIV_a(),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DA+UIV-Pi': lambda: UIV_Pi(),
        'DA+UIV': lambda: UIV(),
        'DA+IV': lambda: IV(),
        'IRM': lambda: LevelCV(
            estimator=IRM(model='linear'),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'AR': lambda: CV(
            estimator=AR(),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'V-REx': lambda: LevelCV(
            estimator=VREx(model='linear'),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'MM-REx': lambda: LevelCV(
            estimator=MMREx(model='linear'),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'RICE': lambda: CV(
            estimator=RICE(model='linear'),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'ICP': lambda: ICP(),
        'DRO': lambda: DRO(model='linear')
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}
    
    # sweep over lambda parameter
    status.update(sweep='lambda')
    logger.info('Sweeping over lambda parameters.')
    lambda_values, results = LambdaSweep(
        seed=seed,
        n_samples=n_samples,
        kernel_dim=kernel_dim,
        n_experiments=n_experiments,
        methods=methods,
        sweep_samples=sweep_samples
    ).run_experiment()
    save(
        obj=lambda_values, fname='lambda_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=results, fname='lambda_results', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        lambda_values, results,
        bootstrapped=True, **ANNOTATE_SWEEP_PLOT['lambda']
    )
    sweep_plot(
        lambda_values, results, **ANNOTATE_SWEEP_PLOT['lambda']
    )

    # sweep over gamma parameter
    status.update(sweep='gamma')
    logger.info('Sweeping over gamma parameters.')
    gamma_values, results = GammaSweep(
        seed=seed,
        n_samples=n_samples,
        kernel_dim=kernel_dim,
        n_experiments=n_experiments,
        methods=methods,
        sweep_samples=sweep_samples
    ).run_experiment()
    save(
        obj=gamma_values, fname='gamma_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=results, fname='gamma_results', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        gamma_values, results,
        bootstrapped=True, **ANNOTATE_SWEEP_PLOT['gamma']
    )
    sweep_plot(
        gamma_values, results, **ANNOTATE_SWEEP_PLOT['gamma']
    )

    # sweep over alpha parameter
    status.update(sweep='alpha')
    logger.info('Sweeping over alpha parameters.')
    alpha_values, results = AlphaSweep(
        seed=seed,
        n_samples=n_samples,
        kernel_dim=kernel_dim,
        n_experiments=n_experiments,
        methods=methods,
        sweep_samples=sweep_samples
    ).run_experiment()
    save(
        obj=alpha_values, fname='alpha_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=results, fname='alpha_results', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        alpha_values, results,
        bootstrapped=True, **ANNOTATE_SWEEP_PLOT['alpha']
    )
    sweep_plot(
        alpha_values, results, **ANNOTATE_SWEEP_PLOT['alpha']
    )

    # no sweep, just compare baselines with gamma=1 and lambda=1
    status.update(sweep='N/A')
    logger.info('Linear experiemnt with gamma=1 and lambda=1.')
    _, results = BaselineExperiment(
        seed=seed,
        n_samples=n_samples,
        kernel_dim=kernel_dim,
        n_experiments=n_experiments,
        methods=methods
    ).run_experiment()
    save(
        obj=results, fname=EXPERIMENT, experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=results, fname=EXPERIMENT, experiment=EXPERIMENT, format='json'
    )
    box_plot(
        results, fname=EXPERIMENT, experiment=EXPERIMENT,
        savefig=True, bootstrapped=True, **ANNOTATE_BOX_PLOT[EXPERIMENT]
    )
    box_plot(
        results, fname=EXPERIMENT, experiment=EXPERIMENT,
        savefig=True, **ANNOTATE_BOX_PLOT[EXPERIMENT]
    )
    
    caption = f'RE $\pm$ one std across {n_experiments} experiments of {n_samples} samples each.',
    table = tex_table(
        results, label=EXPERIMENT, caption=caption
    )
    save(
        obj=table, fname=EXPERIMENT, experiment=EXPERIMENT, format='tex'
    )
    table_bootstrapped = tex_table(
        results, label=EXPERIMENT, caption=caption, bootstrapped=True
    )
    save(
        obj=table_bootstrapped,
        fname=EXPERIMENT+'_bootstrapped', experiment=EXPERIMENT,
        format='tex'
    )


if __name__ == '__main__':
    CLI = ArgumentParser(description='Linear simulation experiment.')
    CLI.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    CLI.add_argument(
        '--n_samples', type=int, default=2000, help='Number of samples per experiment.'
    )
    CLI.add_argument('--n_experiments', type=int, default=10, help='Number of experiments.')
    CLI.add_argument(
        '--sweep_samples', type=int, default=10, help='Sweep resolution across lambda, alpha and gamma.'
    )
    CLI.add_argument(
        '--methods',
        nargs="*",
        type=str,
        default=['ERM', 'DA+ERM', 'DA+UIV-CV', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+UIV-CV DA+IV`.'
    )
    args = CLI.parse_args()
    run(**vars(args))
