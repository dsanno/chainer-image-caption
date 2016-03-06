# -*- coding: utf-8 -*-

import argparse
import cPickle as pickle
import json
import numpy as np
import scipy.io
import random
import chainer
from chainer import cuda, Variable, optimizers, serializers, functions as F
from chainer.functions.evaluation import accuracy
from net import ImageCaption
import time

parser = argparse.ArgumentParser(description='Train image caption model')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--sentence', '-s', required=True, type=str,
                    help='input sentences dataset file path')
parser.add_argument('--image', '-i', required=True, type=str,
                    help='input images file path')
parser.add_argument('--model', '-m', default=None, type=str,
                    help='input model and state file path without extension')
parser.add_argument('--output', '-o', required=True, type=str,
                    help='output model and state file path without extension')
parser.add_argument('--iter', default=100, type=int,
                    help='output model and state file path without extension')
args = parser.parse_args()

gpu_device = None
args = parser.parse_args()
xp = np
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu
    cuda.get_device(gpu_device).use()
    xp = cuda.cupy

with open(args.sentence, 'rb') as f:
    sentence_dataset = pickle.load(f)
image_dataset = scipy.io.loadmat(args.image)
images = image_dataset['feats'].transpose((1, 0))

train_image_ids = sentence_dataset['images']['train']
train_sentences = sentence_dataset['sentences']['train']
test_image_ids = sentence_dataset['images']['test']
test_sentences = sentence_dataset['sentences']['test']
word_ids = sentence_dataset['word_ids']
feature_num = images.shape[1]
hidden_num = 1024
batch_size = 20

print 'word count: ', len(word_ids)
caption_net = ImageCaption(len(word_ids), feature_num, hidden_num)
if gpu_device is not None:
    caption_net.to_gpu(gpu_device)
optimizer = optimizers.Adam()
optimizer.setup(caption_net)

if args.model is not None:
    serializers.load_hdf5(args.model + '.model', caption_net)
    serializers.load_hdf5(args.model + '.state', optimizer)

bos = word_ids['<S>']
eof = word_ids['</S>']
unknown = word_ids['<UNK>']

def random_batches(image_groups, sentence_groups):
    batches = []
    for image_ids, sentences in zip(image_groups, sentence_groups):
        length = len(sentences)
        index = np.arange(length, dtype=np.int32)
        np.random.shuffle(index)
        for n in range(0, length, batch_size):
            batch_index = index[n:n + batch_size]
            batches.append((image_ids[batch_index], sentences[batch_index]))
    random.shuffle(batches)
    return batches

def make_groups(image_ids, sentences, train=True):
    if train:
        boundaries = [1, 6, 11, 16, 21, 31, 41]
    else:
        boundaries = range(1, 41)
    sentence_groups = []
    image_groups = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        size = sum(map(lambda x: len(sentences[x]), range(begin, end)))
        sub_sentences = np.full((size, end + 1), eof, dtype=np.int32)
        sub_sentences[:, 0] = bos
        sub_image_ids = np.zeros((size,), dtype=np.int32)
        offset = 0
        for n in range(begin, end):
            length = len(sentences[n])
            if length > 0:
                sub_sentences[offset:offset + length, 1:n + 1] = sentences[n]
                sub_image_ids[offset:offset + length] = image_ids[n]
            offset += length
        sentence_groups.append(sub_sentences)
        image_groups.append(sub_image_ids)
    return image_groups, sentence_groups

def forward(net, image_batch, sentence_batch, train=True):
    images = Variable(xp.asarray(image_batch), volatile=not train)
    n, sentence_length = sentence_batch.shape
    net.initialize(images)
    loss = 0
    acc = 0
    for i in range(sentence_length - 1):
        x = Variable(xp.asarray(sentence_batch[:, i]), volatile=not train)
        t = Variable(xp.asarray(sentence_batch[:, i + 1]), volatile=not train)
        y = net(x, train=train)
        loss += F.softmax_cross_entropy(y, t)
        acc += accuracy.accuracy(y, t)
    return loss, acc

def train(epoch_num):
    image_groups, sentence_groups = make_groups(train_image_ids, train_sentences)
    test_image_groups, test_sentence_groups = make_groups(test_image_ids, test_sentences, train=False)
    for epoch in range(epoch_num):
        batches = random_batches(image_groups, sentence_groups)
        sum_loss = 0
        sum_acc = 0
        sum_size = 0
        batch_num = len(batches)
        for i, (image_id_batch, sentence_batch) in enumerate(batches):
            loss, acc = forward(caption_net, images[image_id_batch], sentence_batch)
            optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            sentence_length = sentence_batch.shape[1]
            sum_loss += float(loss.data) * len(sentence_batch)
            sum_acc += float(acc.data) * len(sentence_batch)
            sum_size += len(sentence_batch) * (sentence_length - 1)
            if (i + 1) % 1000 == 0:
                print '{} / {}'.format(i + 1, batch_num)
        print 'epoch: {} done'.format(epoch + 1)
        print 'train loss: {} accuracy: {}'.format(sum_loss / sum_size, sum_acc / sum_size)
        sum_loss = 0
        sum_acc = 0
        sum_size = 0
        for image_ids, sentences in zip(test_image_groups, test_sentence_groups):
            if len(sentences) == 0:
                continue
            size = len(sentences)
            for i in range(0, size, batch_size):
                image_id_batch = image_ids[i:i + batch_size]
                sentence_batch = sentences[i:i + batch_size]
                loss, acc = forward(caption_net, images[image_id_batch], sentence_batch, train=False)
                sentence_length = sentence_batch.shape[1]
                sum_loss += float(loss.data) * len(sentence_batch)
                sum_acc += float(acc.data) * len(sentence_batch)
                sum_size += len(sentence_batch) * (sentence_length - 1)
        print 'test loss: {} accuracy: {}'.format(sum_loss / sum_size, sum_acc / sum_size)

        serializers.save_hdf5(args.output + '_{0:04d}.model'.format(epoch), caption_net)
        serializers.save_hdf5(args.output + '_{0:04d}.state'.format(epoch), optimizer)

train(args.iter)
