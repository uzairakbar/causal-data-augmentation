import scipy
import enlighten
import numpy as np
import scipy as sp
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.simulation.nonlinear import NonlinearSimulationDA as DA

from src.sem.simulation.nonlinear import NonlinearSimulationSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector

from src.regressors.erm import GradientDescentERM as ERM

from src.regressors.iv import GeneralizedMomentMethodIV as IV

from src.regressors.uiv import GeneralizedMomentMethodUnfaithfulIV as UIV_a

from src.regressors.baselines import (
    RICE,
    MiniMaxREx as MMREx,
    VarianceREx as VREx,
    AnchorRegression as AR,
    InvariantRiskMinimization as IRM,
    DistributionallyRobustOptimization as DRO,
)

from src.regressors.model_selectors import (
    LevelCV,
    VanillaCV as CV,
)

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
DEFAULT_CV_SAMPLES: int=10
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_JOBS: int=1
POLYNOMIAL_DEGREE: int=1
FEATURES = PolynomialFeatures(
    POLYNOMIAL_DEGREE, include_bias=False
)


def run(
        seed: int,
        n_samples: int,
        n_experiments: int,
        methods: List[str],
        augmentations: Optional[List[str]]=[None],
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
    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ERM': lambda: ERM(model='2-layer'),
        'DA+ERM': lambda: ERM(model='2-layer'),
        'DA+UIV-CV': lambda: CV(
            metric='mse',
            estimator=UIV_a(model='2-layer'),
            param_distributions = {
                'alpha': sp.stats.loguniform.rvs(
                    1e-5, 1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DA+UIV-LCV': lambda: LevelCV(
            metric='mse',
            estimator=UIV_a(model='2-layer'),
            param_distributions = {
                'alpha': sp.stats.loguniform.rvs(
                    1e-5, 1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DA+IV': lambda: IV(model='2-layer'),
        'IRM': lambda: LevelCV(
            metric='mse',
            estimator=IRM(model='2-layer'),
            param_distributions = {
                'alpha': sp.stats.loguniform.rvs(
                    1e-5, 1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'AR': lambda: CV(
            metric='mse',
            estimator=AR(model='2-layer'),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'V-REx': lambda: LevelCV(
            metric='mse',
            estimator=VREx(model='2-layer'),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'MM-REx': lambda: LevelCV(
            metric='mse',
            estimator=MMREx(model='2-layer'),
            param_distributions = {
                'alpha': np.random.normal(
                    0, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'RICE': lambda: CV(
            metric='mse',
            estimator=RICE(model='2-layer'),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DRO': lambda: DRO(model='2-layer'),
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}
    
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
            G = FEATURES.fit_transform(G)

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

                pbar_methods.update(), status.update()
            pbar_methods.close()
            pbar_sem.update(), status.update()
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
        all_errors,label=EXPERIMENT,
        caption=f'Test MSE $\\pm$ one standard deviation across {n_experiments} runs.'
    )
    save(
        obj=table, fname=EXPERIMENT, experiment=EXPERIMENT, format='tex'
    )
    
    grid_plot(
        all_functions, fname=EXPERIMENT, experiment=EXPERIMENT, savefig=True
    )

    status.close()


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
        default=['ERM', 'DA+ERM', 'DA+UIV-CV', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+UIV-CV DA+IV`.'
    )
    args = CLI.parse_args()
    run(**vars(args))
