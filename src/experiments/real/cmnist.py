import numpy as np

from src.data_augmentors.real.cmnist import ColoredDigitsDA as DA

from src.sem.real.cmnist import ColoredDigitsSEM as SEM

from src.regressors.erm import LeastSquaresGradientDescent as ERM
from src.regressors.iv import IVGeneralizedMomentMethod as IV
from src.regressors.daiv import DAIVGeneralizedMomentMethod as DAIV

from src.regressors.model_selectors import LeaveOneOut as LOO

from src.experiments.utils import (
    set_seed,
    box_plot,
    tex_table,
)


ALL_METHODS = {
    "ERM": lambda: ERM(model="cmnist"),
    "DA+ERM": lambda: ERM(model="cmnist"),
    "DAIV+LOO": lambda: LOO(
        metric="accuracy",
        estimator=DAIV(model="cmnist"),
        param_distributions = {"alpha": np.random.lognormal(1, 1, 10)},
        cv=5                                # TODO: proper LOO CV
    ),
    "DA+IV": lambda: IV(model="cmnist")
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
    X_test, y_test = sem_test(N = args["n_samples"])
    for i in range(args["num_seeds"]):
        set_seed(i)

        sem = SEM(train=True)
        da = DA()

        X, y = sem(N = args["n_samples"])
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
            else:
                model.fit(X=GX, y=y, Z=G)
            
            y_test_hat = model.predict(X_test)
            all_errors[method_name][i] = accuracy(y_test, y_test_hat)

    return all_errors


def main():
    args = {
        "n_samples": 60000,
        "num_seeds": 10,
        "methods": "ERM,DA+ERM,DAIV+LOO,DA+IV"
    }

    all_errors = run(args)
    box_plot(all_errors, xlabel="accuracy", fname="cmnist")
    tex_table(
        all_errors, fname="cmnist", highlight="max",
        title="Results for the CMNIST experiment with {num_seeds} random seeds.".format(num_seeds=args["num_seeds"])
    )


if __name__ == '__main__':
    main()

