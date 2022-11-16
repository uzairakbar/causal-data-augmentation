import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.simulation import (
    linear as linear_simulation,
    nonlinear as nonlinear_simulation,
)
from src.experiments.real import (
    linear as optical_device_experiment,
    cmnist as cmnist_experiment,
    rmnist as rmnist_experiment,
)


linear_simulation.main()
nonlinear_simulation.main()
optical_device_experiment.main()
cmnist_experiment.main()
rmnist_experiment.main()

