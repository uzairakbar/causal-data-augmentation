import torch
import numpy as np
from torchvision import datasets

from src.sem.abstract import StructuredEquationModel as SEM


# Build environments
def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()


class ColoredDigitsSEM(SEM):
    @staticmethod
    def load_dataset(directory="data/mnist", train=True):
        mnist = datasets.MNIST(directory, train=train, download=True)
        return mnist
    
    _TRAIN = load_dataset.__func__()
    _TEST = load_dataset.__func__(train=False)

    def __init__(self, train=True):
        self.train = train
        if train:
            self.images = self._TRAIN.data
            self.targets = self._TRAIN.targets
        else:
            self.images = self._TEST.data
            self.targets = self._TEST.targets
        return super(ColoredDigitsSEM, self).__init__()
    
    def __len__(self):
        return len(self.images)
    
    def sample(self, N = 1, **kwargs):
        N_max = len(self.images)
        indices = np.arange(N_max)
        replace = N > N_max
        sampled = np.random.choice(indices,
                                   N,
                                   replace)
        images, targets = self.images[sampled], self.targets[sampled]

        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1

        # 2x subsample for computational convenience
        N_X = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        y_ = (targets < 5).float()
        y = torch_xor(y_, torch_bernoulli(0.25, N))
        # Assign a color based on the label; flip the color with probability e
        if self.train:
            e_space = torch.tensor([0.1, 0.2])
        else:
            e_space = torch.tensor([0.9])
        idx = torch.multinomial(e_space, num_samples=N, replacement=True)
        e = e_space[idx]
        colors = torch_xor(y, torch_bernoulli(e, N))
        # Apply the color to the image by zeroing out the other color channel
        X_zeros = torch.zeros_like(N_X)
        X = torch.stack([N_X, N_X, X_zeros], dim=1)
        X[torch.tensor(range(N)), (1 - colors).long(), :, :] *= 0
        return ((X.float() / 255.).numpy(),
                y[:, None].numpy())

