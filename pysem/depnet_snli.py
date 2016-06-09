import platform
import random
import time

import numpy as np

from handlers import SNLI
from networks import MLP, DependencyNetwork

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
snli.load_vocab('snli_words')

train_data = [d for d in snli.train_data if d[1] != '-']
dev_data = [d for d in snli.dev_data if d[1] != '-']

dim = 200
iters = 50
rate = 0.01
batchsize = 2500

s1_depnet = DependencyNetwork(embedding_dim=dim, vocab=snli.vocab)
s2_depnet = DependencyNetwork(embedding_dim=dim, vocab=snli.vocab)

classifier = MLP(di=2*dim, dh=1000, do=3)


def compute_accuracy(data):
    count = 0
    for sample in data:
        s1 = sample[0][0]
        s2 = sample[0][1]
        label = sample[1]

        s1_depnet.forward_pass(s1)
        s2_depnet.forward_pass(s2)

        bias = np.ones(1).reshape(1, 1)
        s1 = s1_depnet.get_sentence_embedding()
        s2 = s2_depnet.get_sentence_embedding()

        xs = np.concatenate((bias, s1, s2))
        prediction = classifier.predict(xs)
        if prediction == label:
            count += 1

    print('Dev set accuracy: ', count / float(len(data)))


def train(iters, batchsize, rate):
    for _ in range(iters):
        batch = random.sample(train_data, batchsize)
        print('On training iteration ', _)

        if _ % 5 == 0 and _ != 0:
            rate = rate / 2.0
            print('Dropped rate to ', rate)

        for sample in batch:
            s1 = sample[0][0]
            s2 = sample[0][1]
            label = sample[1]

            s1_depnet.forward_pass(s1)
            s2_depnet.forward_pass(s2)

            bias = np.ones(1).reshape(1, 1)
            emb1 = s1_depnet.get_sentence_embedding()
            emb2 = s2_depnet.get_sentence_embedding()

            xs = np.concatenate((bias, emb1, emb2))
            ys = label

            classifier.train(xs, ys, iters=1, rate=rate)

            emb1_grad = classifier.yi_grad[1:dim+1]
            emb2_grad = classifier.yi_grad[dim+1:]

            s1_depnet.backward_pass(emb1_grad, rate=rate)
            s2_depnet.backward_pass(emb2_grad, rate=rate)

        compute_accuracy(dev_data)


start_time = time.time()

train(iters, batchsize, rate)

print('Total runtime: ', time.time() - start_time)
