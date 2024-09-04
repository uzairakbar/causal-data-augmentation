import enlighten
import numpy as np
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List

from src.data_augmentors.real.cmnist import ColoredDigitsDA as DA

from src.sem.real.cmnist import ColoredDigitsSEM as SEM

from src.regressors.abstract import Regressor, ModelSelector
from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV
from src.regressors.daiv import DAIVGeneralizedMomentMethod as UIV_a
from src.regressors.daiv import MinMaxDAIV as UIV
from src.regressors.daiv import DAIVConstrainedOptimizationGMM as UIV

from src.regressors.model_selectors import VanillaCV as CV

from src.experiments.utils import (
    save,
    set_seed,
    bootstrap,
    box_plot,
    tex_table,
    fit_model
)

ModelBuilder = Callable[[Optional[float]], Regressor | ModelSelector]
ALL_METHODS: Dict[str, ModelBuilder] = {
    'ERM': lambda: ERM(
        model='cmnist', epochs=40
    ),
    'DA+ERM': lambda: ERM(
        model='cmnist', epochs=40
    ),
    'DA+UIV-5fold': lambda: CV(
        metric='accuracy',
        estimator=UIV_a(
            model='cmnist', gmm_steps=4, epochs=10
        ),
        param_distributions = {'alpha': np.random.lognormal(1, 1, 10)},
        frac=0.2,
        n_jobs=-1,
    ),
    'DA+IV': lambda: IV(
        model='cmnist', gmm_steps=4, epochs=10
    )
}
manager = enlighten.get_manager()


def run(
        seed: int,
        num_seeds: int,
        n_samples: int,
        methods: List[str],
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
    ):
    
    status = manager.status_bar(
        status_format=u'Colored MNIST experiment{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER,
        autorefresh=True, min_delta=0.5
    )

    methods: Dict[str, ModelBuilder] = {m: ALL_METHODS[m] for m in methods}
    
    error_dim = (num_seeds,)
    all_errors = {name: np.zeros(error_dim) for name in methods}
    
    accuracy = lambda y, yhat: (y == yhat).mean()
    sem_test = SEM(train=False)
    X_test, y_test = sem_test(N = n_samples)

    pbar_experiment = manager.counter(
        total=num_seeds, desc='Experiments', unit='experiments'
    )
    for i in range(num_seeds):
        if seed >= 0: set_seed(seed+i)

        sem = SEM(train=True)
        da = DA()

        X, y = sem(N = n_samples)
        GX, G = da(X)

        pbar_methods = manager.counter(
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
                pbar_manager=manager
            )
            
            y_test_hat = model.predict(X_test)
            all_errors[method_name][i] = accuracy(y_test, y_test_hat)

            save(obj=all_errors, fname='cmnist', format='json')

            pbar_methods.update()
        pbar_methods.close()
        pbar_experiment.update()
        status.update()
    pbar_experiment.close()
    
    save(obj=all_errors, fname='cmnist', format='json')
    save(
        obj=np.arange(seed, seed+num_seeds), fname='cmnist_seeds', format='json'
    )

    box_plot(all_errors, xlabel='accuracy', fname='cmnist')
    tex_table(
        all_errors, fname='cmnist', highlight='max',
        caption=f'Test accuracy $\pm$ one standard deviation for the CMNIST experiment across {num_seeds} seeds.'
    )


if __name__ == '__main__':
    CLI = ArgumentParser(description='Colored MNIST experiment.')
    CLI.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    CLI.add_argument(
        '--num_seeds',
        type=int,
        default=10,
        help='Number of seeds to try -- average results over [`seed`, `seed+num_seeds`] seeds.'
    )
    CLI.add_argument(
        '--n_samples',
        type=int,
        default=-1,
        help='Number of samples per experiment. Negative is all available samples.'
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
