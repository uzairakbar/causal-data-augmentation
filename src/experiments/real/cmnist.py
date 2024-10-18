import enlighten
import numpy as np
from loguru import logger
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.real.cmnist import ColoredDigitsDA as DA

from src.sem.real.cmnist import ColoredDigitsSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector

from src.regressors.erm import GradientDescentERM as ERM

from src.regressors.iv import GeneralizedMomentMethodIV as IV

from src.regressors.uiv import GeneralizedMomentMethodUnfaithfulIV as UIV_a

from src.regressors.baselines import (
    RICE,
    MiniMaxREx as MMREx,
    VarianceREx as VREx,
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
    box_plot,
    bootstrap,
    tex_table,
    fit_model,
    ANNOTATE_BOX_PLOT,
)


ModelBuilder = Callable[[Optional[float]], Regressor | ModelSelector]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='colored_mnist'
DEFAULT_CV_SAMPLES: int=10
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_JOBS: int=1
POLYNOMIAL_DEGREE: int=1
FEATURES = PolynomialFeatures(
    POLYNOMIAL_DEGREE, include_bias=False
)


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
        'DA+UIV-CV': lambda: CV(
            metric='accuracy',
            estimator=UIV_a(model='cmnist'),
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
            metric='accuracy',
            estimator=UIV_a(model='cmnist'),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DA+IV': lambda: IV(model='cmnist'),
        'IRM': lambda: LevelCV(
            metric='accuracy',
            estimator=IRM(model='cmnist'),
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
            metric='accuracy',
            estimator=VREx(model='cmnist'),
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
            metric='accuracy',
            estimator=MMREx(model='cmnist'),
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
            metric='accuracy',
            estimator=RICE(model='cmnist'),
            param_distributions = {
                'alpha': np.random.exponential(
                    1, getattr(cv, 'samples', DEFAULT_CV_SAMPLES)
                )
            },
            frac=getattr(cv, 'frac', DEFAULT_CV_FRAC),
            n_jobs=getattr(cv, 'n_jobs', DEFAULT_CV_JOBS),
            verbose=1
        ),
        'DRO': lambda: DRO(model='cmnist'),
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}

    all_accuracies = {
        augmentation: {
            name: np.zeros(num_seeds) for name in methods
        } for augmentation in augmentations
    }

    accuracy = lambda y, yhat: 100*(y == yhat).mean()
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
            G = FEATURES.fit_transform(G)
            
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
                    # pbar_manager=MANAGER,
                    da=da
                )
                
                y_test_hat = model.predict(X_test)
                score = accuracy(y_test, y_test_hat)
                all_accuracies[augmentation][method_name][i] = score

                logger.info(f'Test accuracy for {method_name}: \t {score}.')
                
                pbar_methods.update(), status.update()
            pbar_methods.close()
            pbar_experiment.update(), status.update()
        pbar_experiment.close()
        pbar_augmentation.update(), status.update()
    pbar_augmentation.close()

    save(
        obj=all_accuracies, fname=EXPERIMENT, experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=np.arange(seed, seed+num_seeds),
        fname='seeds', experiment=EXPERIMENT, format='json'
    )
    
    box_plot(
        all_accuracies, xlabel='Test Accuracy (%)',
        fname=EXPERIMENT, experiment=EXPERIMENT, savefig=True,
        **ANNOTATE_BOX_PLOT[EXPERIMENT]
    )
    
    caption = (
        f'Test accuracy $\pm$ one standard deviation for the CMNIST experiment across {num_seeds} seeds.'
    )
    table = tex_table(
        all_accuracies, label=EXPERIMENT, highlight='max', caption=caption
    )
    save(
        obj=table, fname=EXPERIMENT, experiment=EXPERIMENT, format='tex'
    )


if __name__ == '__main__':
    CLI = ArgumentParser(
        description='Colored-MNIST experiment.'
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
        default=['ERM', 'DA+ERM', 'DA+UIV-CV', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+UIV-CV DA+IV`.'
    )
    args = CLI.parse_args()
    run(**vars(args))
