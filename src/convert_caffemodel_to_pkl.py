import cPickle as pickle
from chainer.links import caffe
import sys

model_path = sys.argv[1]
pkl_path = sys.argv[2]

model = caffe.CaffeFunction(model_path)
with open(pkl_path, 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
