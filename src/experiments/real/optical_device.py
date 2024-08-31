import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Callable, Optional

from src.data_augmentors.real.optical_device import OpticalDeviceDA as DA

from src.sem.real.optical_device import OpticalDeviceSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector
from src.regressors.erm import LeastSquaresClosedForm as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV
# from src.regressors.daiv import DAIVGeneralizedMomentMethod as DAIV
# from src.regressors.daiv import MinMaxDAIV as mmDAIV
# from src.regressors.daiv import DAIVProjectedLeastSquaresClosedForm as pDAIV
from src.regressors.daiv import DAIVProjectedLeastSquares as DAIVpi
from src.regressors.daiv import DAIVLeastSquaresClosedForm as DAIValpha
from src.regressors.daiv import DAIVConstrainedLeastSquares as DAIV
from src.regressors.iv import IVTwoStageLeastSquares as IV

from src.regressors.model_selectors import LeaveOneOut as LOO
from src.regressors.model_selectors import LeaveOneLevelOut as LOLO
from src.regressors.model_selectors import ConfounderCorrection as CC

from src.experiments.utils import (
    save,
    set_seed,
    relative_sq_error,
    bootstrap,
    box_plot,
    tex_table,
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
    # 'DAIV+LOLO': lambda: LOLO(
    #     estimator=DAIValpha(),
    #     param_distributions = {'alpha': np.random.lognormal(1, 1, 10)},
    #     n_jobs=-1,
    # ),
    'DAIV+CC': lambda: CC(estimator=DAIValpha()),
    # 'mmDAIV': lambda: mmDAIV(),
    # 'DAIVP_': lambda: pDAIV(),
    'DAIVpi': lambda: DAIVpi(),
    'DAIV': lambda: DAIV(),
    'DA+IV': lambda: IV()
}


def run(
        seed: int,
        n_samples: int,
        alpha: float,
        methods: str
    ):
    if seed >= 0:
        set_seed(seed)
    
    if methods == 'all':
        methods = ALL_METHODS
    else:
        methods = {m: ALL_METHODS[m] for m in methods.split(',')}
    
    all_sems = []
    all_augmenters = []
    
    for exp in range(SEM.num_experiments()):
        sem = SEM(exp)
        da = DA()
        all_sems.append(sem)
        all_augmenters.append(da)
    
    error_dim = (SEM.num_experiments(),)
    all_errors = {name: np.zeros(error_dim) for name in methods}
    
    for j, (sem, da) in enumerate(tqdm(
            zip(all_sems, all_augmenters), total=SEM.num_experiments(), desc='Experiments'
        )):
        sem_solution = sem.solution

        X, y = sem(N = n_samples)
        GX, G = da(X)
        for method_name, method in (pbar_methods := tqdm(
                methods.items(), total=len(methods), desc='Methods'
            )):
            pbar_methods.set_description(f'{method_name}')

            model = method()
            if 'ERM' in method_name:
                if 'DA' in method_name:
                    model.fit(X=GX, y=y)
                else:
                    model.fit(X=X, y=y)
            elif 'DAIV' in method_name:
                model.fit(X=X, y=y, G=G, GX=GX)
            else:
                model.fit(X=X, y=y, Z=G)
            
            method_solution = model.solution
            
            error = relative_sq_error(sem_solution, method_solution)

            all_errors[method_name][j] = error
    
    errors_bootstrapped = bootstrap(all_errors)
    box_plot(errors_bootstrapped, fname='optical_device')
    tex_table(
        errors_bootstrapped, fname='optical_device',
        caption='RSE $\pm$ one standard deviation across the optical device datasets.'
    )
    save(all_errors, 'optical_device')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optical device dataset experiment.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    parser.add_argument(
        '--n_samples', type=int, default=1000, help='Number of samples per experiment.'
    )
    parser.add_argument('--alpha', type=float, default=2.0, help='UIValpha regressoin parameter.')
    parser.add_argument(
        '--methods',
        type=str,
        default='all',
        help='Methods to use. Specify in comma-separated format -- "ERM,DA+ERM,DA+UIV,DA+IV". Default is "all".'
    )
    args = parser.parse_args()
    run(**args)
