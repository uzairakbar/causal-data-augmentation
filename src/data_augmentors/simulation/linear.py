import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from typing import Dict, Literal, List, Optional

from src.data_augmentors.abstract import DataAugmenter as DA


class NullSpaceTranslation(DA):
    def __init__(self, W_XY: NDArray):
        self.W_ZXtilde = self.null_space(W_XY.T).T
        self.param_dimension, _ = self.W_ZXtilde.shape
    
    @property
    def augmentation(self):
        return 'translate'
    
    def augment(
            self, X: NDArray, gamma: float=1.0
        ) -> Tuple[NDArray, NDArray]:
        N = len(X)
        G = np.random.randn(N, self.param_dimension)

        GX = X + gamma * G @ self.W_ZXtilde
        
        return GX, G
    
    @staticmethod
    def null_space(
            W: NDArray,
            absolute_tolerance: float=1e-13,
            relative_tolerance: float=0.0
        ) -> NDArray:
        U, s, VT = np.linalg.svd(W)
        
        max_singular = s[0]
        tolerance = max(absolute_tolerance,
                        relative_tolerance * max_singular)
        
        num_singular = (s >= tolerance).sum()
        null_space_basis = VT[num_singular:].T
        
        return null_space_basis


Augmentation = Literal['translate']

class LinearSimulationDA(DA):
    def __init__(
            self,
            W_XY: NDArray,
            augmentations: Optional[str]=None
        ):
        all_augmentations: Dict[Augmentation, DA] = {
            augmenter.augmentation: augmenter for augmenter in ([
                NullSpaceTranslation(W_XY=W_XY),
            ])
        }
        
        if augmentations:
            augmentations: List[Augmentation] = augmentations.replace(' ','').split('>')
        else:
            augmentations: List[Augmentation] = list(all_augmentations.keys())
        
        self._augmentations: List[DA] = ([
            all_augmentations[augmentation] for augmentation in augmentations
        ])

    @property
    def augmentation(self):
        return 'linear_simulation'
    
    def augment(
            self,
            X: NDArray
        ) -> Tuple[NDArray, NDArray]:

        GX: NDArray = X.copy()
        G_list: List[NDArray] = []
        for i, augmentation in enumerate(self._augmentations):
            GX, G = augmentation(GX)
            print(f'{augmentation.augmentation} {G.shape}')
            G_list.append(G)
        G: NDArray = np.hstack(G_list)
        
        return GX, G
