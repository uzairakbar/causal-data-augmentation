import argparse
import numpy as np

from src.data_augmentors.real.optical_device import OpticalDeviceDA as DA

from src.sem.real.optical_device import OpticalDeviceSEM as SEM

from src.regressors.erm import LeastSquaresClosedForm as ERM
# from src.regressors.iv import IVGeneralizedMomentMethod as IV
# from src.regressors.daiv import DAIVGeneralizedMomentMethod as DAIV
from src.regressors.daiv import MinMaxDAIV as mmDAIV
from src.regressors.daiv import DAIVProjectedLeastSquares as pDAIV
from src.regressors.daiv import DAIVLeastSquaresClosedForm as DAIV
from src.regressors.iv import IVTwoStageLeastSquares as IV

from src.regressors.model_selectors import LeaveOneOut as LOO
from src.regressors.model_selectors import LeaveOneLevelOut as LOLO
from src.regressors.model_selectors import ConfounderCorrection as CC

from src.experiments.utils import (
    set_seed,
    relative_sq_error,
    box_plot,
    tex_table,
)


ALL_METHODS = {
    "ERM": lambda: ERM(),
    "DA+ERM": lambda: ERM(),
    "DAIV+LOO": lambda: LOO(
        estimator=DAIV(),
        param_distributions = {"alpha": np.random.lognormal(1, 1, 10)},
        cv=5                                # TODO: proper LOO CV
    ),
    "DAIV+LOLO": lambda: LOLO(
        estimator=DAIV(),
        param_distributions = {"alpha": np.random.lognormal(1, 1, 10)}
    ),
    "DAIV+CC": lambda: CC(estimator=DAIV()),
    "mmDAIV": lambda: mmDAIV(),
    "pDAIV": lambda: pDAIV(),
    "DA+IV": lambda: IV()
}

def run(args):
    if args["seed"] >= 0:
        set_seed(args["seed"])
    
    if args["methods"] == "all":
        methods = ALL_METHODS
    else:
        methods = {m: ALL_METHODS[m] for m in args["methods"].split(',')}
    
    all_sems = []
    all_augmenters = []
    
    for exp in range(SEM.num_experiments()):
        sem = SEM(exp)
        da = DA()
        all_sems.append(sem)
        all_augmenters.append(da)
    
    error_dim = (SEM.num_experiments(),)
    all_errors = {name: np.zeros(error_dim) for name in methods}
    
    for j, (sem, da) in enumerate(zip(all_sems, all_augmenters)):
        sem_solution = sem.solution

        X, y = sem(N = args["n_samples"])
        GX, G = da(X)
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
            
            method_solution = model.solution
            
            error = relative_sq_error(sem_solution, method_solution)

            all_errors[method_name][j] = error

    return all_errors


def main():
    parser = argparse.ArgumentParser(description='Optical Device Dataset')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)  # Negative is random
    parser.add_argument('--methods', type=str, default="ERM,DA+ERM,DAIV+LOO,DAIV+LOLO,DAIV+CC,mmDAIV,pDAIV,DA+IV")
    parser.add_argument('--alpha', type=float, default=2.0)
    args = dict(vars(parser.parse_args()))

    all_errors = run(args)
    box_plot(all_errors, fname="optical_device")
    tex_table(
        all_errors, fname="optical_device",
        title="Results for the optical device experiment."
    )


if __name__ == '__main__':
    main()

