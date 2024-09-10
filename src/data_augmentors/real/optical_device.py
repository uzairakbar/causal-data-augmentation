import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from typing import Dict, Literal, List, Optional, Tuple

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
        
        return GX, self.param_scaler(G).reshape(-1,1)
    
    @property
    def augmentation(self):
        return 'permutation'
    
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
    @property
    def augmentation(self):
        return 'rotation'
    
    def augment(self, x, g):
        ROTATION90 = np.array([6, 3, 0,
                               7, 4, 1,
                               8, 5, 2])
        return self.permute(x, g, ROTATION90)

class RandomHorizontalFlip(RandomPermutation):
    @property
    def augmentation(self):
        return 'hflip'
    
    def augment(self, x, g):
        HORIZONTAL_FLIP = np.array([2, 1, 0,
                                    5, 4, 3,
                                    8, 7, 6])
        return self.permute(x, g, HORIZONTAL_FLIP)

class RandomVerticalFlip(RandomPermutation):
    @property
    def augmentation(self):
        return 'vflip'
    
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
    
    @property
    def augmentation(self):
        return 'gaussian_noise'
    
    def augment(self, X, G):
        return X + np.sqrt(0.1) * np.std(X) * G


Augmentation = (Literal[
    'rotation', 'hflip', 'vflip', 'gaussian_noise'
])
ALL_AUGMENTATIONS: Dict[Augmentation, DA] = {
    augmenter.augmentation: augmenter for augmenter in ([
        RandomRotation(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        GaussianNoise()
    ])
}


class OpticalDeviceDA(DA):
    def __init__(
            self,
            augmentations: Optional[str]=None
        ):
        if augmentations:
            augmentations: List[Augmentation] = augmentations.replace(' ','').split('>')
        else:
            augmentations: List[Augmentation] = list(ALL_AUGMENTATIONS.keys())
        
        self._augmentations: List[DA] = ([
            ALL_AUGMENTATIONS[augmentation] for augmentation in augmentations
        ])

    @property
    def augmentation(self):
        return 'optical_device'
    
    def augment(
            self,
            X: NDArray
        ) -> Tuple[NDArray, NDArray]:

        GX: NDArray = X.copy()
        G_list: List[NDArray] = []
        for i, augmentation in enumerate(self._augmentations):
            GX, G = augmentation(GX)
            G_list.append(G)
        G: NDArray = np.hstack(G_list)

        return GX, G
