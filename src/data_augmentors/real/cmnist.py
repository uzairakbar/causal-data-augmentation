import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F

from src.data_augmentors.abstract import DataAugmenter as DA


class UniformStandardScaler():
    def __init__(self, low, high):
        self.mean = (low + high)/2.0
        self.std = (high - low)/(12.0**0.5)
    
    def __call__(self, sample):
        return (sample - self.mean)/self.std


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
        self.param_scaler = UniformStandardScaler(-1*hue, hue)
        return super(Hue, self).__init__(0, 0, 0, hue = hue)

    def forward(self, img):
        _, _, _, _, param = \
            self.get_params(None, None, None, self.hue)

        if param is not None:
            img = F.adjust_hue(img, param)
        
        scaled_param = self.param_scaler(param)
        return img, (scaled_param,)

class Translate(transforms.RandomAffine):
    def __init__(self, translate = (0.0, 0.0)):
        dx, dy = translate
        self.param_scaler = (UniformStandardScaler(-1*dx, dx),
                             UniformStandardScaler(-1*dy, dy))
        return super(Translate, self).__init__(0,
                                               translate=translate,
                                               scale=None,
                                               shear=None,
                                               interpolation=transforms.InterpolationMode.BILINEAR,
                                               fill=0)
    
    def forward(self, img):    
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
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

