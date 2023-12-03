import os
import sys
sys.path.append("/Users/uzair/Documents/github/uzairakbar/daiv")


import numpy as np

from src.data_augmentors.real.cmnist import ColoredDigitsDA as DA

from src.sem.real.cmnist import ColoredDigitsSEM as SEM

from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV
from src.regressors.daiv import DAIVGeneralizedMomentMethod as DAIV
from src.regressors.daiv import MinMaxDAIV as mmDAIV
from src.regressors.daiv import DAIVConstrainedOptimizationGMM as DAIVp

# from src.regressors.model_selectors import LeaveOneOut as LOO
from src.regressors.model_selectors import VanillaCV as CV

from src.experiments.utils import (
    set_seed,
    box_plot,
    tex_table,
)


ALL_METHODS = {
    "ERM": lambda: ERM(
        model="cmnist", epochs=1000
    ),
    "DA+ERM": lambda: ERM(
        model="cmnist", epochs=1000
    ),
    "DAIV+LOO": lambda: CV(
        metric="accuracy",
        estimator=DAIV(
            model="cmnist", gmm_steps=10, epochs=100
        ),
        param_distributions = {"alpha": np.random.lognormal(1, 1, 10)},
        frac=0.2,
        n_jobs=-1,
    ),
    "DAIV100": DAIV(
        model="cmnist", alpha=100, gmm_steps=10, epochs=100
    ),
    "DAIV10": DAIV(
        model="cmnist", alpha=10, gmm_steps=10, epochs=100
    ),
    "DAIV": lambda: mmDAIV(
        model="cmnist", epochs=1000
    ),
    "DAIVp": lambda: DAIVp(
        model="cmnist", epochs=1000
    ),
    "DA+IV": lambda: IV(
        model="cmnist", gmm_steps=10, epochs=100
    ),
    "IV": lambda: IV(
        model="cmnist", gmm_steps=10, epochs=100
    ),
}

def run(args):
    if args["methods"] == "all":
        methods = ALL_METHODS
    else:
        methods = {m: ALL_METHODS[m] for m in args["methods"].split(',')}
    
    error_dim = (args["num_seeds"],)
    all_errors = {name: np.zeros(error_dim) for name in methods}
    
    accuracy = lambda y, yhat: (y == yhat).mean()
    # TODO: -1 for all samples
    sem_test = SEM(train=False)
    X_test, y_test, _ = sem_test(N = args["n_samples"])
    for i in range(args["num_seeds"]):
        set_seed(i)

        sem = SEM(train=True)
        da = DA()

        X, y, z = sem(N = args["n_samples"])
        GX, G = da(X)

        for method_name, method in methods.items():
            print(f"######### {i} {method_name} #########")
            model = method()
            if "ERM" in method_name:
                if "DA" in method_name:
                    model.fit(X=GX, y=y)
                else:
                    model.fit(X=X, y=y)
            elif "DAIV" in method_name:
                model.fit(X=X, y=y, G=G, GX=GX)
            elif method_name == "IV":
                model.fit(X=X, y=y, Z=z)
            else:
                model.fit(X=GX, y=y, Z=G)
            
            y_test_hat = model.predict(X_test)
            all_errors[method_name][i] = accuracy(y_test, y_test_hat)

    return all_errors


def main():
    args = {
        "n_samples": 60000,
        "num_seeds": 1,
        "methods": "all"
    }

    all_errors = run(args)
    box_plot(all_errors, xlabel="accuracy", fname="cmnist")
    tex_table(
        all_errors, fname="cmnist", highlight="max",
        title="Results for the CMNIST experiment with {num_seeds} random seeds.".format(num_seeds=args["num_seeds"])
    )


import json
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

results = []

def run_parallel(seed, n_samples=60000, methods="all"):
    if methods == "all":
        methods = ALL_METHODS
    else:
        methods = {m: ALL_METHODS[m] for m in args["methods"].split(',')}
    
    all_errors = {name: 0.0 for name in methods}
    
    accuracy = lambda y, yhat: (y == yhat).mean()
    # TODO: -1 for all samples
    sem_test = SEM(train=False)
    X_test, y_test, _ = sem_test(N = n_samples)
    
    set_seed(seed)

    sem = SEM(train=True)
    da = DA()

    X, y, z = sem(N = n_samples)
    GX, G = da(X)

    for method_name, method in methods.items():
        print(f"######### {seed} {method_name} #########")
        model = method()
        if "ERM" in method_name:
            if "DA" in method_name:
                model.fit(X=GX, y=y)
            else:
                model.fit(X=X, y=y)
        elif "DAIV" in method_name:
            model.fit(X=X, y=y, G=G, GX=GX)
        elif method_name == "IV":
                model.fit(X=X, y=y, Z=z)
        else:
            model.fit(X=GX, y=y, Z=G)
        
        y_test_hat = model.predict(X_test)
        all_errors[method_name] = accuracy(y_test, y_test_hat)
        all_errors[method_name] = np.random.randn()

    return (seed, all_errors)


if __name__ == '__main__':
    args = {
        "n_samples": 60000,
        "num_seeds": 2,
        "methods": "all"
    }
    # processes = []
    # for i in range(args["num_seeds"]):
    #     processes.append(
    #         Process(
    #             target=run_parallel,
    #             args=(i, args["n_samples"], args["methods"])
    #         )
    #     )
    # for i in range(args["num_seeds"]):
    #     processes[i].start()

    with Pool(args["num_seeds"]) as p:
        results = p.map(
            run_parallel,
            list(range(args["num_seeds"]))
        )
    
    print(results)

    with open('assets/cmnist.json', 'w') as fp:
        json.dump(results, fp)

