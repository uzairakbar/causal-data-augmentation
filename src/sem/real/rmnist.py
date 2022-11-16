import torch
import numpy as np
from torchvision import datasets

from src.sem.abstract import StructuredEquationModel as SEM

from src.data_augmentors.real.rmnist import Rotation


# Build environments
def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()


class RotatedDigitsSEM(SEM):
    @staticmethod
    def load_dataset(directory="data/mnist", train=True):
        mnist = datasets.MNIST(directory, train=train, download=True)
        return mnist
    
    _TRAIN = load_dataset.__func__()
    _TEST = load_dataset.__func__(train=False)
    _ROTATIONS = {
        0: Rotation(0.0),
        30: Rotation(30.0),
        60: Rotation(60.0),
        90: Rotation(90.0),
    }

    def __init__(self, target_rotation=0):
        self.target_rotation = target_rotation
        
        self.train_images = self._TRAIN.data
        self.train_targets = self._TRAIN.targets
        self.test_images = self._TEST.data
        self.test_targets = self._TEST.targets
        return super(RotatedDigitsSEM, self).__init__()
    
    def __len__(self):
        return len(self.train_images)
    
    @classmethod
    def num_rotations(cls):
        return len(cls._ROTATIONS)
    
    @classmethod
    def get_rotator(cls, angle):
        return cls._ROTATIONS[angle]
    
    @classmethod
    def get_rotations(cls):
        return list(cls._ROTATIONS.keys())
    
    def sample(self, N = 1, train = True, **kwargs):
        if train:
            return self._sample_train(N = N)
        else:
            return self._sample_test(N = N)
    
    def _sample_train(self, N = 1):
        N_max = len(self.train_images)
        indices = np.arange(N_max)
        replace = N > N_max
        sampled = np.random.choice(indices,
                                   N,
                                   replace)
        images, targets = self.train_images[sampled], self.train_targets[sampled]

        X = images.reshape((-1, 28, 28))
        y = targets
        
        # Assign a color based on the label; flip the color with probability e
        train_rotations = [rotation for rotation in self._ROTATIONS if rotation != self.target_rotation]
        r_map = {
            i: r for (i, r) in zip(
                range(10),
                np.random.choice(train_rotations, size=10) 
            )
        }

        for i in range(10):
            angle = r_map[i]
            r = self.get_rotator(angle)
            X[y == i], _ = r(X[y == i])

        return (
            X[:, None, :, :].float().numpy(),
            y[:, None].numpy()
        )

    def _sample_test(self, N = 1):
        N_max = len(self.test_images)
        indices = np.arange(N_max)
        replace = N > N_max
        sampled = np.random.choice(indices,
                                   N,
                                   replace)
        images, targets = self.test_images[sampled], self.test_targets[sampled]

        X = images.reshape((-1, 28, 28))
        y = targets

        r = self.get_rotator(self.target_rotation)
        X, _ = r(X)
        
        return (
            X.float().numpy(),
            y[:, None].numpy()
        )