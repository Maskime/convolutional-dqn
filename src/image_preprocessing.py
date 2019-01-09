# Image Preprocessing

# Importing the libraries
import numpy as np
from gym.core import ObservationWrapper
from gym.spaces.box import Box
from gym.wrappers import TimeLimit
from scipy.misc import imresize, imsave


# Preprocessing the Images

class ImageSize:

    def __init__(self, w: int, h: int) -> None:
        super().__init__()
        self.w = w
        self.h = h

    @staticmethod
    def from_str(to_parse: str):
        w, h = to_parse.split('x')
        return ImageSize(w=int(w), h=int(h))


class PreprocessImage(ObservationWrapper):

    def __init__(self, env: TimeLimit, image_size: ImageSize, grayscale=True, crop=lambda img: img, with_crop=True,
                 with_color=True):
        super(PreprocessImage, self).__init__(env)
        self.img_size = (image_size.h, image_size.w)
        self.grayscale = grayscale
        self.crop = crop
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, image_size.h, image_size.w], dtype=np.float32)
        self.frame = 1
        self.do_crop = with_crop
        self.do_color = with_color

    def observation(self, img):
        # 45 50 184
        background = np.array([45, 50, 184])
        # 80 0 132
        border = np.array([80, 0, 132])
        if self.do_crop:
            img = self.crop(img)
        if self.do_color:
            img = np.where(img == background, [0, 0, 0], img)
            img = np.where(np.logical_and(img != border, img != [0, 0, 0]), [254, 0, 0], img)
        self.frame += 1
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.
        return img
