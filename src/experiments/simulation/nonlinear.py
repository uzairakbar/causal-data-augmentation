import scipy
import enlighten
import numpy as np
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List

from src.data_augmentors.simulation.nonlinear import NonlinearSimulationDA as DA

from src.sem.simulation.nonlinear import NonlinearSimulationSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector

from src.regressors.iv import IVGeneralizedMomentMethod as IV

from src.regressors.erm import LeastSquaresGradientDescent as ERM

from src.regressors.daiv import DAIVGeneralizedMomentMethod as UIV_a
from src.regressors.daiv import DAIVConstrainedLeastSquares as UIV

from src.regressors.baselines import RICE as RICE
from src.regressors.baselines import MiniMaxREx as MMREx
from src.regressors.baselines import VarianceREx as VREx
from src.regressors.baselines import AnchorRegression as AR
from src.regressors.baselines import InvariantRiskMinimization as IRM
from src.regressors.baselines import DistributionallyRobustOptimization as DRO

from src.regressors.model_selectors import LeaveOneOut as KFold
from src.regressors.model_selectors import LeaveOneLevelOut as LOLO


from src.experiments.utils import (
    save,
    set_seed,
    grid_plot,
    tex_table,
    fit_model
)


ModelBuilder = Callable[[Optional[float]], Regressor | ModelSelector]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='nonlinear_simulation'
DEFAULT_CV_SAMPLES: int=5
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_FOLDS: int=5
DEFAULT_CV_JOBS: int=1


def run(
        seed: int,
        n_samples: int,
        n_experiments: int,
        methods: List[str],
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
    ):
    status = MANAGER.status_bar(
        status_format=u'Non-linear simulation{fill}Function {function}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, function='<function>',
        autorefresh=True, min_delta=0.5
    )

    if seed >= 0: set_seed(seed)
    
    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ERM': lambda: ERM(model='2-layer'),
        'DA+ERM': lambda: ERM(model='2-layer'),
        'DA+UIV-5fold': lambda: KFold(
            metric='mse',
            estimator=UIV_a(model='2-layer'),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            cv=getattr(cv, 'folds', DEFAULT_CV_FOLDS),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS)
        ),
        'DA+UIV-LOLO': lambda: LOLO(
            metric='mse',
            estimator=UIV_a(model='2-layer'),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS)
        ),
        'DA+IV': lambda: IV(model='2-layer'),
        'IRM': lambda: LOLO(
            metric='mse',
            estimator=IRM(model='cmnist'),
            param_distributions = {
                'alpha': scipy.stats.loguniform.rvs(
                    1e-5, 1e-1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        )
    }
    methods: ModelBuilder = {m: all_methods[m] for m in methods}
    
    all_sems = []
    all_augmenters = []
    for function in SEM.get_functions():
        sem = SEM(function)
        da = DA(function)
        all_sems.append(sem)
        all_augmenters.append(da)
    
    mse = lambda y, yhat: ((y - yhat)**2).mean()
    all_functions = {
        func: {
            name: np.zeros((100, n_experiments)) for name in methods
        } for func in SEM.get_functions()
    }
    all_errors = {
        func: {
            name: np.zeros(n_experiments) for name in methods
        } for func in SEM.get_functions()
    }
    x_gt = np.linspace(-5, 5, 100).reshape(-1, 1)
    for j, (sem, da) in enumerate(zip(all_sems, all_augmenters)):
        status.update(function=sem.function_name)

        all_functions[sem.function_name]['x'] = x_gt.flatten()
        all_functions[sem.function_name]['y'] = sem.f(x_gt).flatten()

        pbar_sem = MANAGER.counter(
            total=n_experiments, desc=f'Function {sem.function_name}', unit='experiments'
        )
        for i in range(n_experiments):
        
            X, y = sem(N = n_samples)
            GX, G = da(X)

            X_test, _ = sem(N = n_samples)
            y_test = sem.f(X_test)

            if 'x_data' in all_functions[sem.function_name]:
                all_functions[sem.function_name]['x_data'] = np.append(
                    all_functions[sem.function_name]['x_data'], X.flatten()
                )
                all_functions[sem.function_name]['y_data'] = np.append(
                    all_functions[sem.function_name]['y_data'], y.flatten()
                )
            else:
                all_functions[sem.function_name]['x_data'] = X.flatten()
                all_functions[sem.function_name]['y_data'] = y.flatten()
            
            pbar_methods = MANAGER.counter(
                total=len(methods), desc=f'Experiment {i}', unit='methods', leave=False
            )
            for method_name, method in methods.items():

                model = method()
                fit_model(
                    model=model,
                    name=method_name,
                    X=X, y=y, G=G, GX=GX,
                    hyperparameters=hyperparameters,
                    pbar_manager=MANAGER
                )

                y_hat = model.predict(x_gt)
                
                all_functions[sem.function_name][method_name][:, i] = (
                    y_hat.flatten()
                )
                
                y_test_hat = model.predict(X_test)
                all_errors[sem.function_name][method_name][i] = (
                    mse(y_test, y_test_hat)
                )

                pbar_methods.update()
            pbar_methods.close()
            pbar_sem.update()
        pbar_sem.close()
    
    save(
        obj=all_errors, fname=f'{EXPERIMENT}_errors',
        experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=all_functions, fname=f'{EXPERIMENT}_functions',
        experiment=EXPERIMENT, format='pkl'
    )
    
    table = tex_table(
        all_errors,
        caption='Test MSE $\\pm$ one standard deviation across $10$ runs.'
    )
    save(
        obj=table, fname=EXPERIMENT, experiment=EXPERIMENT, format='tex'
    )
    
    grid_plot(
        all_functions, fname=EXPERIMENT, experiment=EXPERIMENT, savefig=True
    )


if __name__ == '__main__':
    CLI = ArgumentParser(
        description='Nonlinear simulation experiment.'
    )
    CLI.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    CLI.add_argument(
        '--n_samples', type=int, default=1000, help='Number of samples per experiment.'
    )
    CLI.add_argument('--n_experiments', type=int, default=10, help='Number of experiments.')
    CLI.add_argument(
        '--methods',
        nargs="*",
        type=str,
        default=['ERM', 'DA+ERM', 'DA+UIV-5fold', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+UIV-5fold DA+IV`.'
    )
    args = CLI.parse_args()
    run(**vars(args))
