import os
import sys
import configparser
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.simulation import (
    linear as linear_simulation,
    nonlinear as nonlinear_simulation,
)
from src.experiments.real import (
    optical_device as optical_device_experiment,
    cmnist as cmnist_experiment,
    rmnist as rmnist_experiment,
)

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')

    logger.info('Running linear simulation experiment.')
    linear_simulation.run(
        seed = config.getint('linear simulation', 'seed'),
        n_samples = config.getint('linear simulation', 'n_samples'),
        x_dimension = config.getint('linear simulation', 'x_dimension'),
        n_experiments = config.getint('linear simulation', 'n_experiments'),
        sweep_samples = config.getint('linear simulation', 'sweep_samples'),
        methods = config.get('linear simulation', 'methods')
    )

    logger.info('Running non-linear simulation experiment.')
    nonlinear_simulation.run(
        seed = config.getint('nonlinear simulation', 'seed'),
        n_samples = config.getint('nonlinear simulation', 'n_samples'),
        n_experiments = config.getint('nonlinear simulation', 'n_experiments'),
        methods = config.get('nonlinear simulation', 'methods')
    )

    logger.info('Running optical device experiment.')
    optical_device_experiment.run(
        seed = config.getint('optical device', 'seed'),
        n_samples = config.getint('optical device', 'n_samples'),
        x_dimension = config.getint('optical device', 'x_dimension'),
        n_experiments = config.getint('optical device', 'n_experiments'),
        sweep_samples = config.getint('optical device', 'sweep_samples'),
        methods = config.get('optical device', 'methods')
    )

    logger.info('Running colored MNIST experiment.')
    cmnist_experiment.run(
        seed = config.getint('colored mnist', 'seed'),
        num_seeds = config.getint('colored mnist', 'num_seeds'),
        n_samples = config.getint('colored mnist', 'n_samples'),
        methods = config.get('colored mnist', 'methods')
    )

    logger.info('Running rotated MNIST experiment.')
    rmnist_experiment.run(
        seed = config.getint('rotated mnist', 'seed'),
        num_seeds = config.getint('rotated mnist', 'num_seeds'),
        n_samples = config.getint('rotated mnist', 'n_samples'),
        methods = config.get('rotated mnist', 'methods')
    )
