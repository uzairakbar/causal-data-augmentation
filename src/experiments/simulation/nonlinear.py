import argparse
import enlighten
import numpy as np
from enlighten import Manager
from typing import Dict, Callable, Optional

from src.data_augmentors.simulation.nonlinear import NonlinearSimulationDA as DA

from src.sem.simulation.nonlinear import NonlinearSimulationSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector
from src.regressors.iv import IVGeneralizedMomentMethod as IV
from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.daiv import DAIVGeneralizedMomentMethod as DAIV
from src.regressors.daiv import MinMaxDAIV as mmDAIV

from src.regressors.model_selectors import LeaveOneOut as LOO
from src.regressors.model_selectors import LeaveOneLevelOut as LOLO

from src.experiments.utils import (
    save,
    set_seed,
    grid_plot,
    tex_table,
)


ALL_METHODS: Dict[str, Callable[[Optional[float]], Regressor | ModelSelector]] = {
    'ERM': lambda: ERM(model='2-layer'),
    'DA+ERM': lambda: ERM(model='2-layer'),
    'DAIV+LOO': lambda: LOO(
        metric='mse',
        estimator=DAIV(model='2-layer'),
        param_distributions = {'alpha': np.random.lognormal(1, 1, 10)},
        cv=5,                                # TODO: proper LOO CV
        n_jobs=-1,
    ),
    'DAIV+LOLO': lambda: LOLO(
        metric='mse',
        estimator=DAIV(model='2-layer'),
        param_distributions = {'alpha': np.random.lognormal(1, 1, 10)},
        n_jobs=-1,
    ),
    'mmDAIV': lambda: mmDAIV(model='2-layer'),
    'DA+IV': lambda: IV(model='2-layer')
}
MANAGER = enlighten.get_manager()


def run(
        seed: int,
        n_samples: int,
        n_experiments: int,
        methods: str,
        manager: Manager=MANAGER
    ):
    status = manager.status_bar(
        status_format=u'Non-linear simulation{fill}Function {function}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, function='<function>',
        autorefresh=True, min_delta=0.5
    )

    if seed >= 0:
        set_seed(seed)
    
    if methods == 'all':
        methods = ALL_METHODS
    else:
        methods = {m: ALL_METHODS[m] for m in methods.split(',')}
    
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

        pbar_sem = manager.counter(
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
            
            pbar_methods = manager.counter(
                total=len(methods), desc=f'Experiment {i}', unit='methods', leave=False
            )
            for method_name, method in methods.items():

                model = method()
                if 'ERM' in method_name:
                    if 'DA' in method_name:
                        model.fit(X=GX, y=y, pbar_manager=manager)
                    else:
                        model.fit(X=X, y=y, pbar_manager=manager)
                elif 'DAIV' in method_name:
                    model.fit(X=GX, y=y, G=G, GX=GX, pbar_manager=None)
                else:
                    model.fit(X=GX, y=y, Z=G, pbar_manager=manager)

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
    
    grid_plot(all_functions, fname='nonlinear_simulation')
    tex_table(
        all_errors, fname='nonlinear_simulation',
        caption='Test MSE $\\pm$ one standard deviation across $10$ runs.'
    )
    save(all_errors, 'nonlinear_sim_errors')
    save(all_functions, 'nonlinear_sim_functions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nonlinear simulation experiment.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    parser.add_argument(
        '--n_samples', type=int, default=1000, help='Number of samples per experiment.'
    )
    parser.add_argument('--n_experiments', type=int, default=10, help='Number of experiments.')
    parser.add_argument(
        '--methods',
        type=str,
        default='all',
        help='Methods to use. Specify in comma-separated format -- "ERM,DA+ERM,DA+UIV,DA+IV". Default is "all".'
    )
    args = parser.parse_args()
    run(**vars(args))
