from src.experiments.simulation import (
    linear as linear_simulation,
    nonlinear as nonlinear_simulation
)
from src.experiments.real import (
    linear as optical_device_experiment
)


linear_simulation.main()
nonlinear_simulation.main()
optical_device_experiment.main()

