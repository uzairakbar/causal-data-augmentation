import enlighten
import numpy as np
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List

from src.data_augmentors.real.optical_device import OpticalDeviceDA as DA

from src.sem.real.optical_device import OpticalDeviceSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector
from src.regressors.erm import LeastSquaresClosedForm as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV
# from src.regressors.daiv import DAIVGeneralizedMomentMethod as UIV_a
# from src.regressors.daiv import DAIVProjectedLeastSquaresClosedForm as UIV_Pi
from src.regressors.daiv import DAIVProjectedLeastSquares as UIV_Pi
from src.regressors.daiv import DAIVLeastSquaresClosedForm as UIV_a
from src.regressors.daiv import DAIVConstrainedLeastSquares as UIV
from src.regressors.iv import IVTwoStageLeastSquares as IV

from src.regressors.model_selectors import LeaveOneOut as KFold
from src.regressors.model_selectors import LeaveOneLevelOut as LOLO
from src.regressors.model_selectors import ConfounderCorrection as CC

from src.experiments.utils import (
    save,
    set_seed,
    relative_sq_error,
    bootstrap,
    box_plot,
    tex_table,
    fit_model
)


ModelBuilder = Callable[[Optional[float]], Regressor | ModelSelector]
MANAGER = enlighten.get_manager()
DEFAULT_CV_SAMPLES=10
DEFAULT_CV_FOLDS=5
DEFAULT_CV_JOBS=1


def run(
        seed: int,
        n_samples: int,
        methods: List[str],
        augmentations: Optional[List[str]],
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
    ):
    status = MANAGER.status_bar(
        status_format=u'Optical device experiment{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER,
        autorefresh=True, min_delta=0.5
    )

    if seed >= 0: set_seed(seed)
    
    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ERM': lambda: ERM(),
        'DA+ERM': lambda: ERM(),
        'DA+UIV-5fold': lambda: KFold(
            estimator=UIV_a(),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            cv=getattr(cv, 'folds', DEFAULT_CV_FOLDS),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'DA+UIV-LOLO': lambda: LOLO(
            estimator=UIV_a(),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'DA+UIV-CC': lambda: CC(estimator=UIV_a()),
        'DA+UIV-Pi': lambda: UIV_Pi(),
        'DA+UIV': lambda: UIV(),
        'DA+IV': lambda: IV()
    }
    methods: ModelBuilder= {m: all_methods[m] for m in methods}
    
    all_sems = []
    all_augmenters = []
    
    for exp in range(SEM.num_experiments()):
        sem = SEM(exp)
        da = DA()
        all_sems.append(sem)
        all_augmenters.append(da)
    
    error_dim = (SEM.num_experiments(),)
    all_errors = {name: np.zeros(error_dim) for name in methods}
    
    pbar_experiment = MANAGER.counter(
        total=SEM.num_experiments(), desc='Experiments', unit='experiments'
    )
    for j, (sem, da) in enumerate(zip(all_sems, all_augmenters)):
        sem_solution = sem.solution

        X, y = sem(N = n_samples)
        GX, G = da(X)
        
        pbar_methods = MANAGER.counter(
            total=len(methods), desc=f'SEM {j}', unit='methods', leave=False
        )
        for method_name, method in methods.items():

            model = method()
            fit_model(
                model=model,
                name=method_name,
                X=X, y=y, G=G, GX=GX
            )
            
            method_solution = model.solution
            
            error = relative_sq_error(sem_solution, method_solution)

            all_errors[method_name][j] = error
            
            pbar_methods.update()
        pbar_methods.close()
        pbar_experiment.update()
        status.update()
    pbar_experiment.close()
    
    errors_bootstrapped = bootstrap(all_errors)
    box_plot(errors_bootstrapped, fname='optical_device')
    tex_table(
        errors_bootstrapped, fname='optical_device',
        caption='RSE $\pm$ one standard deviation across the optical device datasets.'
    )
    save(all_errors, 'optical_device')


if __name__ == '__main__':
    CLI = ArgumentParser(
        description='Optical device dataset experiment.'
    )
    CLI.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    CLI.add_argument(
        '--n_samples', type=int, default=1000, help='Number of samples per experiment.'
    )
    CLI.add_argument(
        '--methods',
        nargs="*",
        type=str,
        default=['ERM', 'DA+ERM', 'DA+UIV-5fold', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+UIV-5fold DA+IV`.'
    )
    args = CLI.parse_args()
    run(**vars(args))
