import torch
import numpy as np
from numbers import Number
from torchvision import transforms
from torchvision.transforms import functional as F

from src.data_augmentors.abstract import DataAugmenter as DA


class Compose(transforms.Compose):
    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)

    def __call__(self, img):
        params = tuple()
        for t in self.transforms:
            img, transform_params = t(img)
            params += transform_params
        return img, params


class Rotation(transforms.RandomRotation):
    def __init__(self, degrees):
        if isinstance(degrees, Number):
            degrees = (degrees, degrees)
        return super(Rotation, self).__init__(
            degrees = degrees
        )

    def forward(self, img):
        angle = self.get_params(self.degrees)
        return (
            F.rotate(img, angle),
            (
                np.sin((np.pi/180.0)*angle),
                np.cos((np.pi/180.0)*angle)
            )
        )


class Flip(transforms.RandomHorizontalFlip):
    def __init__(self, degrees):
        return super(Flip, self).__init__()
    
    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img), (-1,)
        return img, (1,)


class RotatedDigitsDA(DA):
    def __init__(self):
        self._augmentor = Compose([
            # Flip(),
            Rotation(degrees = (0, 360))
        ])
    
    def augment(self, X):
        GX, G = [], []
        for image in X:
            gx, g = self._augmentor(torch.tensor(image))
            GX.append(gx.numpy())
            G.append(np.array(g))
        GX = np.stack(GX)
        G = np.stack(G)
        return GX, G
