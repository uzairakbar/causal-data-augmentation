import torch
import numpy as np
from torchvision import transforms
from abc import ABC, abstractmethod
from torchvision.transforms import functional as F

from src.data_augmentors.abstract import DataAugmenter as DA


class StandardScaler():
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, sample):
        return (sample - self.mean)/self.std


class UniformStandardScaler(StandardScaler):
    def __init__(self, low, high):
        self.mean = (low + high)/2.0
        self.std = (high - low)/(12.0**0.5)


ALPHA, BETA = 1, 1
class BetaStandardScaler(StandardScaler):
    def __init__(self, min, max, a=ALPHA, b=BETA):
        self.mean = a/(a + b)
        self.std = np.sqrt(a*b)/((a+b) * np.sqrt(a+b+1))
        self.min = min
        self.max = max
    
    def rescale(self, sample):
        return sample*(self.max-self.min) + self.min
    
    def __call__(self, sample):
        sample = (sample-self.min)/(self.max-self.min)
        return super(BetaStandardScaler, self).__call__(sample=sample)


class Compose(transforms.Compose):
    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)

    def __call__(self, img):
        params = tuple()
        for t in self.transforms:
            img, transform_params = t(img)
            params += transform_params
        return img, params


class Hue(transforms.ColorJitter):
    def __init__(self, hue = 0):
        self.param_scaler = BetaStandardScaler(-1*hue, hue)
        return super(Hue, self).__init__(0, 0, 0, hue = hue)
    
    def get_params(
        self
    ):
        h = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        h = self.param_scaler.rescale(h)
        return h

    def forward(self, img):
        param = self.get_params()
        img = F.adjust_hue(img, param)
        scaled_param = self.param_scaler(param)
        return img, (scaled_param,)


class Translate(transforms.RandomAffine):
    def __init__(self, translate = (0.0, 0.0)):
        dx, dy = translate
        self.param_scaler = (BetaStandardScaler(-1*dx, dx),
                             BetaStandardScaler(-1*dy, dy))
        return super(Translate, self).__init__(0,
                                               translate=translate,
                                               scale=None,
                                               shear=None,
                                               interpolation=transforms.InterpolationMode.BILINEAR,
                                               fill=0)
    def get_params(
        self,
        degrees,
        img_size,
    ):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        
        tx = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        ty = float( torch.distributions.beta.Beta(ALPHA, BETA).sample() )
        tx = self.param_scaler[0].rescale(tx)
        ty = self.param_scaler[1].rescale(ty)
        tx = int(round(tx * img_size[0]))
        ty = int(round(tx * img_size[1]))
        translations = (tx, ty)

        scale = 1.0
        shear_x = shear_y = 0.0
        shear = (shear_x, shear_y)

        return angle, translations, scale, shear
    
    def forward(self, img):    
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(img)

        ret = self.get_params(self.degrees, img_size)
        params = ret[1]
        
        img = F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
        scaled_params = tuple(param_scaler(param) for (param_scaler, param) in zip(self.param_scaler, params))
        return img, scaled_params


class ColoredDigitsDA(DA):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode='RGB')
        self.to_tensor = transforms.ToTensor()
        self._augmentor = Compose([Hue(0.5), Translate((0.2, 0.2))])
    
    def augment(self, X):
        GX, G = [], []
        for image in X:
            gx, g = self._augmentor(torch.tensor(image))
            GX.append(gx.numpy())
            G.append(np.array(g))
        GX = np.stack(GX)
        G = np.stack(G)
        return GX, G
