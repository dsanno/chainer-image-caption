import os
import cPickle as pickle
import numpy as np
from PIL import Image
import chainer
from chainer import cuda, functions as F
from chainer.links import caffe

class ImageModel(object):
    def __init__(self):
        self.image_shape = self._image_shape()
        self.mean_image = self._mean_image()
        self.func = None

    def feature(self, image_path):
        xp = self.func.xp
        array = xp.asarray(self.load_image(image_path))
        x = xp.ascontiguousarray(array)
        return self._feature(x)

    def load(self, path):
        root, ext = os.path.splitext(path)
        if ext == '.pkl':
            with open(path, 'rb') as f:
                self.func = pickle.load(f)
        else:
            self.func = caffe.CaffeFunction(path)

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_w, image_h = self.image_shape
        w, h = image.size
        if w > h:
            shape = (image_w * w / h, image_h)
        else:
            shape = (image_w, image_h * h / w)
        x = (shape[0] - image_w) / 2
        y = (shape[1] - image_h) / 2
        pixels = np.asarray(image.resize(shape).crop((x, y, x + image_w, y + image_h))).astype(np.float32)
        pixels = pixels[:,:,::-1].transpose(2,0,1)
        pixels -= self.mean_image
        return pixels.reshape((1,) + pixels.shape)

    def to_gpu(self, device=None):
        self.func.to_gpu()

    def to_cpu(self):
        self.func.to_cpu()

    def _image_shape(self):
        raise NotImplementedError

    def _mean_image(self):
        raise NotImplementedError

    def _feature(self, x):
        raise NotImplementedError

class VGG19(ImageModel):
    def __init__(self):
        super(VGG19, self).__init__()

    def _image_shape(self):
        return (224, 224)

    def _mean_image(self):
        mean_image = np.ndarray((3, 224, 224), dtype=np.float32)
        mean_image[0] = 104
        mean_image[1] = 117
        mean_image[2] = 124
        return mean_image

    def _feature(self, x):
        y, = self.func(inputs={'data': x}, outputs=['fc7'])
        return y
