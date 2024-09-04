import os
import sys
import yaml
from loguru import logger
from munch import munchify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.simulation import (
    linear as linear_simulation,
    nonlinear as nonlinear_simulation,
)
from src.experiments.real import (
    optical_device as optical_device_experiment,
    cmnist as colored_mnist_experiment,
    rmnist as rotated_mnist_experiment,
)

if __name__ == '__main__':

    with open('src/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = munchify(config)

    if config.linear_simulaton:
        logger.info('Running linear simulation experiment.')
        linear_simulation.run(
            **config.linear_simulaton,
            **config.hyperparameters
        )
    
    if config.nonlinear_simulaton:
        logger.info('Running non-linear simulation experiment.')
        nonlinear_simulation.run(
            **config.nonlinear_simulaton,
            **config.hyperparameters
        )
    
    if config.optical_device:
        logger.info('Running optical device experiment.')
        optical_device_experiment.run(
            **config.optical_device,
            **config.hyperparameters
        )
    
    if config.colored_mnist:
        logger.info('Running colored MNIST experiment.')
        colored_mnist_experiment.run(
            **config.colored_mnist,
            **config.hyperparameters
        )
    
    if config.rotated_mnist:
        logger.info('Running rotated MNIST experiment.')
        rotated_mnist_experiment.run(
            **config.rotated_mnist,
            **config.hyperparameters
        )
