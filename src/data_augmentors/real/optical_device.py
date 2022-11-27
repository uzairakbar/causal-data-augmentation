import numpy as np
from abc import abstractmethod

from src.data_augmentors.abstract import DataAugmenter as DA


class RandomPermutation(DA):
    def __call__(self, X):
        PERMUTE = ([1, -1])
        
        N = len(X)
        G = np.random.choice(PERMUTE, size=N)
        GX = np.zeros_like(X)
        for i in range(len(X)):
            x = X[i, :]
            g = G[i]
            GX[i, :] = self.augment(x, g)
        
        return GX, G
    
    @abstractmethod
    def augment(self, x, g):
        pass

    @staticmethod
    def permute(x, g, permutation):
        if g == 1:
            gx = x[permutation]
        elif g == -1:
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


class OpticalDeviceDA(DA):
    def augment(self, X):
        augmentations = ([
            RandomRotation(),
            RandomHorizontalFlip(),
            RandomVerticalFlip()
        ])
        GX = X.copy()
        G = np.zeros(( len(X), len(augmentations) ))
        for i, augmentation in enumerate(augmentations):
            GX, G[:, i] = augmentation(GX)
        
        return GX, G

