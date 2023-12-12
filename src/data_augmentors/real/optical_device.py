import numpy as np
from abc import abstractmethod

from src.data_augmentors.abstract import DataAugmenter as DA

class StandardScaler():
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, sample):
        return (sample - self.mean)/self.std

P = 0.50
class BernoulliStandardScaler(StandardScaler):
    def __init__(self, p=P):
        self.mean = p
        self.std = np.sqrt(p*(1.0-p))


class RandomPermutation(DA):
    def __init__(self, p=P):
        self.p = p
        self.param_scaler = BernoulliStandardScaler(p=p)
        return super(RandomPermutation, self).__init__()

    def __call__(self, X):
        PERMUTE = ([1.0, 0.0])
        
        N = len(X)
        G = np.random.choice(
            PERMUTE, size=N, p=[self.p, (1.0-self.p)]
        )
        GX = np.zeros_like(X)
        for i in range(len(X)):
            x = X[i, :]
            g = G[i]
            GX[i, :] = self.augment(x, g)
        
        return GX, self.param_scaler(G)
    
    @abstractmethod
    def augment(self, x, g):
        pass

    @staticmethod
    def permute(x, g, permutation):
        if g == 1.0:
            gx = x[permutation]
        elif g == 0.0:
            gx = x
        return gx


class RandomRotation(RandomPermutation):
    def augment(self, x, g):
        ROTATION90 = np.array([6, 3, 0,
                               7, 4, 1,
                               8, 5, 2])
        return self.permute(x, g, ROTATION90)

class RandomHorizontalFlip(RandomPermutation):
    def augment(self, x, g):
        HORIZONTAL_FLIP = np.array([2, 1, 0,
                                    5, 4, 3,
                                    8, 7, 6])
        return self.permute(x, g, HORIZONTAL_FLIP)

class RandomVerticalFlip(RandomPermutation):
    def augment(self, x, g):
        VERTICAL_FLIP = np.array([6, 7, 8,
                                  3, 4, 5,
                                  0, 1, 2])
        return self.permute(x, g, VERTICAL_FLIP)


class GaussianNoise(DA):
    def __call__(self, X):
        N, M = X.shape
        G = np.random.randn(N, M)
        GX = self.augment(X, G)
        return GX, G
    
    def augment(self, X, G):
        return X + np.sqrt(0.1) * np.std(X) * G


class OpticalDeviceDA(DA):
    def augment(self, X):
        augmentations = ([
            RandomRotation(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
        ])
        GX = X.copy()
        G = np.zeros(( len(X), len(augmentations) ))
        for i, augmentation in enumerate(augmentations):
            GX, G[:, i] = augmentation(GX)
        
        # # don't use LOLO when using additive Gaussian noise DA
        # noise = GaussianNoise()
        # GX, G_noise = noise(GX)
        # G = np.hstack((G, G_noise))

        return GX, G

