import enlighten
import numpy as np
import scipy as sp
from loguru import logger
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List

from src.data_augmentors.real.cmnist import ColoredDigitsDA as DA

from src.sem.real.cmnist import ColoredDigitsSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector

from src.regressors.erm import LeastSquaresGradientDescent as ERM

from src.regressors.iv import IVGeneralizedMomentMethod as IV

from src.regressors.daiv import DAIVGeneralizedMomentMethod as UIV_a

from src.regressors.baselines import RICE as RICE
from src.regressors.baselines import MiniMaxREx as MMREx
from src.regressors.baselines import VarianceREx as VREx
from src.regressors.baselines import AnchorRegression as AR
from src.regressors.baselines import InvariantRiskMinimization as IRM
from src.regressors.baselines import DistributionallyRobustOptimization as DRO

from src.regressors.model_selectors import LevelCV, VanillaCV as CV


from src.experiments.utils import (
    save,
    set_seed,
    bootstrap,
    box_plot,
    tex_table,
    fit_model
)


ModelBuilder = Callable[[Optional[float]], Regressor | ModelSelector]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='colored_mnist'
DEFAULT_CV_SAMPLES: int=5
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_JOBS: int=1


def run(
        seed: int,
        num_seeds: int,
        n_samples: int,
        methods: List[str],
        augmentations: Optional[List[str]]=[None],
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
    ):

    status = MANAGER.status_bar(
        status_format=u'Colored MNIST experiment{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER,
        autorefresh=True, min_delta=0.5
    )

    if seed >= 0: set_seed(seed)

    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ERM': lambda: ERM(model='cmnist'),
        'DA+ERM': lambda: ERM(model='cmnist'),
        'DA+UIV-5fold': lambda: CV(
            metric='accuracy',
            estimator=UIV_a(model='cmnist'),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'DA+UIV-LOLO': lambda: LevelCV(
            metric='accuracy',
            estimator=UIV_a(model='cmnist'),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
        ),
        'DA+IV': lambda: IV(model='cmnist'),
        'IRM': lambda: LevelCV(
            metric='accuracy',
            estimator=IRM(model='cmnist'),
            param_distributions = {
                'alpha': sp.stats.loguniform.rvs(
                    1e-5, 1e-1, size=getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'AR': lambda: LevelCV(
            metric='accuracy',
            estimator=AR(model='cmnist'),
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
            metric='accuracy',
            estimator=VREx(),
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
            metric='accuracy',
            estimator=MMREx(),
            param_distributions = {
                'alpha': np.random.normal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'RICE': lambda: CV(
            metric='accuracy',
            estimator=RICE(),
            param_distributions = {
                'alpha': np.random.lognormal(
                    1, 1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DRO': lambda: DRO(),
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}

    all_errors = {
        augmentation: {
            name: np.zeros(num_seeds) for name in methods
        } for augmentation in augmentations
    }

    accuracy = lambda y, yhat: (y == yhat).mean()
    sem_test = SEM(train=False)
    X_test, y_test = sem_test(N = -1)

    pbar_augmentation = MANAGER.counter(
        total=len(augmentations), desc='Augmentations', unit='augmentations'
    )
    for augmentation in augmentations:
        
        pbar_experiment = MANAGER.counter(
            total=num_seeds, desc=augmentation, unit='experiments',
        )
        for i in range(num_seeds):
            if seed >= 0: set_seed(seed+i)

            sem = SEM(train=True)
            da = DA(augmentation)

            X, y = sem(N = n_samples)
            GX, G = da(X)
            
            pbar_methods = MANAGER.counter(
                total=len(methods), desc=f'Seed {seed+i}', unit='methods', leave=False
            )
            for method_name, method in methods.items():
                if seed >= 0: set_seed(seed+i)

                model = method()
                fit_model(
                    model=model,
                    name=method_name,
                    X=X, y=y, G=G, GX=GX,
                    hyperparameters=hyperparameters,
                    pbar_manager=MANAGER,
                    da=da
                )
                
                y_test_hat = model.predict(X_test)
                score = accuracy(y_test, y_test_hat)
                all_errors[augmentation][method_name][i] = score

                logger.info(f'Test accuracy for {method_name}: \t {score}.')

                save(
                    obj=all_errors, fname=EXPERIMENT, experiment=EXPERIMENT, format='json'
                )
                
                pbar_methods.update(), status.update()
            pbar_methods.close()
            pbar_experiment.update(), status.update()
        pbar_experiment.close()
        pbar_augmentation.update(), status.update()
    pbar_augmentation.close()

    save(
        obj=all_errors, fname=EXPERIMENT, experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=all_errors, fname=EXPERIMENT, experiment=EXPERIMENT, format='json'
    )
    save(
        obj=np.arange(seed, seed+num_seeds),
        fname='seeds', experiment=EXPERIMENT, format='json'
    )
    
    box_plot(
        all_errors, xlabel='Accuracy',
        fname=EXPERIMENT, experiment=EXPERIMENT, savefig=True
    )
    
    table = tex_table(
        all_errors, label=EXPERIMENT,
        caption=f'Test accuracy $\pm$ one standard deviation for the CMNIST experiment across {num_seeds} seeds.'
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
