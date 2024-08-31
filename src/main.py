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
    linear_simulation.run(**config['LinearSimulation'])

    logger.info('Running non-linear simulation experiment.')
    nonlinear_simulation.run(**config['NonlinearSimulation'])

    logger.info('Running optical device experiment.')
    optical_device_experiment.run(**config['OpticalDevice'])

    logger.info('Running colored MNIST experiment.')
    cmnist_experiment.run(**config['ColoredMNIST'])

    logger.info('Running rotated MNIST experiment.')
    rmnist_experiment.run(**config['RotatedMNIST'])
