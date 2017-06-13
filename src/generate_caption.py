import argparse
import numpy as np
import cPickle as pickle
from image_model import VGG19
from net import ImageCaption

import chainer
from chainer import Variable, serializers, cuda, functions as F

parser = argparse.ArgumentParser(description='Generate image caption')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--image_model', '-i', required=True, type=str,
                    help='input images model file path (*.caffemodel or *.pkl)')
parser.add_argument('--model', '-m', required=True, type=str,
                    help='input model file path')
parser.add_argument('--sentence', '-s', required=True, type=str,
                    help='sentence dataset file path')
parser.add_argument('--list', '-l', required=True, type=str,
                    help='image file list file path')
args = parser.parse_args()

feature_num = 4096
hidden_num = 512
beam_width = 20
max_length = 60

with open(args.sentence, 'rb') as f:
    sentence_dataset = pickle.load(f)
word_ids = sentence_dataset['word_ids']
id_to_word = {}
for k, v in word_ids.items():
    id_to_word[v] = k

image_model = VGG19()
image_model.load(args.image_model)

caption_net = ImageCaption(len(word_ids), feature_num, hidden_num)
serializers.load_hdf5(args.model, caption_net)

xp = np
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    cuda.get_device(gpu_device).use()
    xp = cuda.cupy
    image_model.to_gpu(gpu_device)
    caption_net.to_gpu(gpu_device)

bos = word_ids['<S>']
eos = word_ids['</S>']

with open(args.list) as f:
    paths = filter(bool, f.read().split('\n'))

def generate(net, image_model, image_path):
    feature = image_model.feature(image_path)
    net.initialize(feature)
    candidates = [(net, [bos], 0)]

    for i in range(max_length):
        next_candidates = []
        for prev_net, tokens, likelihood in candidates:
            if tokens[-1] == eos:
                next_candidates.append((None, tokens, likelihood))
                continue
            net = prev_net.copy()
            x = xp.asarray([tokens[-1]]).astype(np.int32)
            y = F.softmax(net(x))
            token_likelihood = np.log(cuda.to_cpu(y.data[0]))
            order = token_likelihood.argsort()[:-beam_width:-1]
            next_candidates.extend([(net, tokens + [i], likelihood + token_likelihood[i]) for i in order])
        candidates = sorted(next_candidates, key=lambda x: -x[2])[:beam_width]
        if all([candidate[1][-1] == eos for candidate in candidates]):
            break
    return [candidate[1] for candidate in candidates]

with chainer.using_config('train', False):
    with chainer.using_config('enable_backprop', False):
        for path in paths:
            sentences = generate(caption_net, image_model, path)
            print '# ', path
            for token_ids in sentences[:5]:
                tokens = [id_to_word[token_id] for token_id in token_ids[1:-1]]
                print ' '.join(tokens)
