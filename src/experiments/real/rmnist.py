import argparse
import numpy as np

from src.data_augmentors.real.rmnist import RotatedDigitsDA as DA

from src.sem.real.rmnist import RotatedDigitsSEM as SEM

from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV
from src.regressors.daiv import DAIVGeneralizedMomentMethod as DAIV

# from src.regressors.model_selectors import LeaveOneOut as LOO
from src.regressors.model_selectors import VanillaCV as CV

from src.experiments.utils import (
    set_seed,
    box_plot,
    tex_table,
)


ALL_METHODS = {
    "ERM": lambda: ERM(model="rmnist"),
    "DA+ERM": lambda: ERM(model="rmnist"),
    "DAIV+LOO": lambda: CV(
        metric="accuracy",
        estimator=DAIV(model="rmnist"),
        param_distributions = {"alpha": np.random.lognormal(1, 1, 10)},
        frac=0.2
    ),
    "DA+IV": lambda: IV(model="rmnist")
}


def run_experiment(args):
    if args["methods"] == "all":
        methods = ALL_METHODS
    else:
        methods = {m: ALL_METHODS[m] for m in args["methods"].split(',')}
    
    all_sems = []
    all_augmenters = []
    for angle in SEM.get_rotations():
        sem = SEM(target_rotation=angle)
        da = DA()
        all_sems.append(sem)
        all_augmenters.append(da)
    
    accuracy = lambda y, yhat: (y == yhat).mean()
    all_errors = {
        fr"${target}^o$": {
            name: np.zeros((args["num_seeds"],)) for name in methods
        } for target in SEM.get_rotations()
    }
    for (sem, da) in zip(all_sems, all_augmenters):
        for i in range(args["num_seeds"]):
            set_seed(i)
            print(f"Angle {sem.target_rotation}, Seed {i}")
        
            X, y = sem(N = args["n_samples"], train = True)
            GX, G = da(X)

            X_test, y_test = sem(N = args["n_samples"], train = False)            

            for method_name, method in methods.items():
                model = method()
                if "ERM" in method_name:
                    if "DA" in method_name:
                        model.fit(X=GX, y=y)
                    else:
                        model.fit(X=X, y=y)
                elif "DAIV" in method_name:
                    model.fit(X=X, y=y, G=G, GX=GX)
                else:
                    model.fit(X=GX, y=y, Z=G)
                
                y_test_hat = model.predict(X_test)
                all_errors[fr"${sem.target_rotation}^o$"][method_name][i] = (
                    accuracy(y_test, y_test_hat)
                )

    return all_errors


def main():
    args = {
        "n_samples": 60000,
        "num_seeds": 10,
        "methods": "ERM,DA+ERM,DAIV+LOO,DA+IV"
    }

    all_errors = run_experiment(args)
    box_plot(all_errors, xlabel="accuracy", fname="rmnist")
    tex_table(
        all_errors, fname="rmnist", highlight="max",
        title="Results for the Rotated MNIST experiment with {num_seeds} random seeds.".format(num_seeds=args["num_seeds"])
    )


if __name__ == '__main__':
    main()

