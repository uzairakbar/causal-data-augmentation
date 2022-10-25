import numpy as np

from src.data_augmentors.abstract import DataAugmenter as DA


class NullSpaceTranslation(DA):
    def __init__(self, W_XY):
        self.W_ZXtilde = self.null_space(W_XY.T).T
        self.param_dimension, _ = self.W_ZXtilde.shape
    
    def augment(self, X, gamma = 1.0):
        N = len(X)
        G = np.random.randn(N, self.param_dimension)

        GX = X + gamma * G @ self.W_ZXtilde
        
        return GX, G
    
    @staticmethod
    def null_space(W,
                   absolute_tolerance=1e-13,
                   relative_tolerance=0):
        U, s, VT = np.linalg.svd(W)
        
        max_singular = s[0]
        tolerance = max(absolute_tolerance,
                        relative_tolerance * max_singular)
        
        num_singular = (s >= tolerance).sum()
        null_space_basis = VT[num_singular:].T
        
        return null_space_basis

