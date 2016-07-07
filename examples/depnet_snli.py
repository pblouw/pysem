import platform
import random
import time
import sys
import numpy as np

from pysem.corpora import SNLI
from pysem.networks import DependencyNetwork
from pysem.utils.ml import MultiLayerPerceptron

if platform.system() == 'Linux':
    snlipath = '/home/pblouw/corpora/snli_1.0/'
    wikipath = '/home/pblouw/corpora/wikipedia'
    cachepath = '/home/pblouw/cache/'
else:
    snlipath = '/Users/peterblouw/corpora/snli_1.0/'
    wikipath = '/Users/peterblouw/corpora/wikipedia'
    cachepath = '/Users/peterblouw/cache/'


snli = SNLI(snlipath)
snli.extractor = snli.get_xy_pairs
snli.load_vocab('snli_words.pickle')

train_data = [d for d in snli.train_data if d[1] != '-']
dev_data = [d for d in snli.dev_data if d[1] != '-']
label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

dim = 200
iters = 100
rate = 0.005
batchsize = 50000

s1_depnet = DependencyNetwork(dim=dim, vocab=snli.vocab)
s2_depnet = DependencyNetwork(dim=dim, vocab=snli.vocab)

classifier = MultiLayerPerceptron(di=2*dim, dh=dim, do=3)


def binarize(label_list):

    label_list = [label_list]
    lookup = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 1}
    y_idx = [lookup[l] for l in label_list]
    x_idx = range(len(label_list))
    vals = np.zeros((3, len(label_list)))
    vals[y_idx, x_idx] = 1
    return vals


def compute_accuracy(data):
    count = 0
    for sample in data:
        s1 = sample[0][0]
        s2 = sample[0][1]
        label = sample[1]

        s1_depnet.forward_pass(s1)
        s2_depnet.forward_pass(s2)

        s1 = s1_depnet.get_root_embedding()
        s2 = s2_depnet.get_root_embedding()

        xs = np.concatenate((s1, s2))
        prediction = classifier.predict(xs)
        if label_dict[prediction[0]] == label:
            count += 1

    print('Dev set accuracy: ', count / float(len(data)))


def train(iters, batchsize, rate):
    for _ in range(iters):
        batch = random.sample(train_data, batchsize)
        print('On training iteration ', _)

        if _ % 10 == 0 and _ != 0:
            rate = rate / 2
            print('Dropped rate to ', rate)

        for sample in batch:
            s1 = sample[0][0]
            s2 = sample[0][1]
            label = sample[1]

            s1_depnet.forward_pass(s1)
            s2_depnet.forward_pass(s2)

            emb1 = s1_depnet.get_root_embedding()
            emb2 = s2_depnet.get_root_embedding()

            xs = np.concatenate((emb1, emb2))
            ys = label

            classifier.train(xs, binarize(ys), rate=rate)

            emb1_grad = classifier.yi_grad[:dim] * s1_depnet.tanh_grad(emb1)
            emb2_grad = classifier.yi_grad[dim:] * s2_depnet.tanh_grad(emb2)

            s1_depnet.backward_pass(emb1_grad, rate=rate)
            s2_depnet.backward_pass(emb2_grad, rate=rate)

        compute_accuracy(dev_data)


start_time = time.time()

train(iters, batchsize, rate)

print('Total runtime: ', time.time() - start_time)
