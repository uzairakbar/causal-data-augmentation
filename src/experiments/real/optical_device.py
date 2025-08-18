import enlighten
import numpy as np
import scipy as sp
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.real.optical_device import OpticalDeviceDA as DA

from src.sem.real.optical_device import OpticalDeviceSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector

from src.regressors.erm import (
    LeastSquaresClosedForm as ERM,
    RidgeRegression,
    LassoRegression,
)

from src.regressors.iv import TwoStageLeastSquaresIV as IV

from src.regressors.ivl import LeastSquaresClosedFormIVlike as IVL_a

from src.regressors.baselines import (
    RICE,
    KaniaWit,
    MiniMaxREx as MMREx,
    VarianceREx as VREx,
    InvariantRiskMinimization as IRM,
    InvariantCausalPrediction as ICP,
    DistributionallyRobustOptimization as DRO,
)

from src.regressors.model_selectors import (
    LevelCV,
    VanillaCV as CV,
    RiskDifference as RD,
    ConfounderCorrection as CC,
)

from src.experiments.utils import (
    save,
    set_seed,
    box_plot,
    tex_table,
    fit_model,
    estimation_error,
    ANNOTATE_BOX_PLOT,
)


ModelBuilder = Callable[[Optional[float]], Regressor | ModelSelector]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='optical_device'
DEFAULT_CV_SAMPLES: int=10
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_FOLDS: int=5
DEFAULT_CV_JOBS: int=1
GROUND_TRUTH: str='polynomial'


def run(
        seed: int,
        n_samples: int,
        methods: List[str],
        augmentations: Optional[List[str]]=[None],
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
        'DA+IVL-CV': lambda: CV(
            estimator=IVL_a(),
            param_distributions = {
                'alpha': sp.stats.loguniform.rvs(
                    1e-4, 1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'DA+IVL-LCV': lambda: LevelCV(
            estimator=IVL_a(),
            param_distributions = {
                'alpha': sp.stats.loguniform.rvs(
                    1e-4, 1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'DA+IVL-CC': lambda: CC(
            estimator=IVL_a(),
            param_distributions = {
                'alpha': sp.stats.loguniform.rvs(
                    1e-4, 1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
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
        'DRO': lambda: DRO(model='linear'),
        'L1Janzing': lambda: CC(
            estimator=LassoRegression(),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'L2Janzing': lambda: CC(
            estimator=RidgeRegression(),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'Kania&Wit': lambda: RD(
            estimator=KaniaWit(),
            param_distributions={
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        )
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}
    
    all_errors = {
        augmentation: {
            name: np.zeros(SEM.num_experiments()) for name in methods
        } for augmentation in augmentations
    }
    all_sems = {
        augmentation: ([
            SEM(
                experiment=exp, ground_truth=GROUND_TRUTH
            ) for exp in range(SEM.num_experiments())
        ]) for augmentation in augmentations
    }
    all_augmenters = {
        augmentation: ([
            DA(augmentation) for exp in range(SEM.num_experiments())
        ]) for augmentation in augmentations
    }

    pbar_augmentation = MANAGER.counter(
        total=len(augmentations), desc='Augmentations', unit='augmentations'
    )
    for augmentation in augmentations:
        pbar_experiment = MANAGER.counter(
            total=SEM.num_experiments(), desc=augmentation, unit='experiments',
        )
        for i, (sem, da) in enumerate(zip(
                all_sems[augmentation], all_augmenters[augmentation]
            )):
            if seed >= 0: set_seed(seed)

            sem_solution = sem.solution
            

            X, y = sem(N = n_samples)
            GX, G = da(X)

            features = PolynomialFeatures(
                sem.poly_degree, include_bias=False
            )
            X = features.fit_transform(X)
            G = features.fit_transform(G)
            GX = features.fit_transform(GX)
            
            pbar_methods = MANAGER.counter(
                total=len(methods), desc=f'SEM {i}', unit='methods', leave=False
            )
            for method_name, method in methods.items():
                if seed >= 0: set_seed(seed)

                model = method()
                fit_model(
                    model=model,
                    name=method_name,
                    X=X, y=y, G=G, GX=GX,
                    hyperparameters=hyperparameters,
                    da=da
                )
                
                method_solution = model.solution
                
                error = estimation_error(sem_solution, method_solution)

                all_errors[augmentation][method_name][i] = error
                
                pbar_methods.update(), status.update()
            pbar_methods.close()
            pbar_experiment.update(), status.update()
        pbar_experiment.close()
        pbar_augmentation.update(), status.update()
    pbar_augmentation.close()

    save(
        obj=all_errors, fname=EXPERIMENT, experiment=EXPERIMENT, format='pkl'
    )
    
    box_plot(
        all_errors, fname=EXPERIMENT, experiment=EXPERIMENT,
        savefig=True, **ANNOTATE_BOX_PLOT[EXPERIMENT]
    )
    
    caption = 'nCER $\pm$ standard error across the optical device datasets.'
    table = tex_table(
        all_errors, label=EXPERIMENT, caption=caption
    )
    save(
        obj=table, fname=EXPERIMENT, experiment=EXPERIMENT, format='tex'
    )


if __name__ == '__main__':
    CLI = ArgumentParser(
        description='Optical device dataset experiment.'
    )
    CLI.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    CLI.add_argument(
        '--n_samples', type=int, default=1_000, help='Number of samples per experiment.'
    )
    CLI.add_argument(
        '--methods',
        nargs="*",
        type=str,
        default=['ERM', 'DA+ERM', 'DA+IVL-CV', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+IVL-CV DA+IV`.'
    )
    args = CLI.parse_args()
    run(**vars(args))
