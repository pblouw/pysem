import platform
import random
import time
import sys
import pickle
import numpy as np

import matplotlib.pyplot as plt

from pysem.corpora import SNLI
from pysem.networks import DependencyNetwork, RecurrentNetwork
from pysem.lstm import LSTM, TreeLSTM
from pysem.utils.ml import MultiLayerPerceptron
from itertools import islice

if platform.system() == 'Linux':
    snlipath = '/home/pblouw/corpora/snli_1.0/'
else:
    snlipath = '/Users/peterblouw/corpora/snli_1.0/'

snli = SNLI(snlipath)
snli.extractor = snli.get_xy_pairs
snli.load_vocab('snli_words.pickle')

# train_data = [d for d in snli.train_data if d.label != '-']
dev_data = [d for d in snli.dev_data if d.label != '-']
label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

# set_size = 2
dim = 100
iters = 50
batchsize = 9000
rate = 0.01

s1_lstm = TreeLSTM(input_dim=dim, cell_dim=dim, vocab=snli.vocab)
s2_lstm = TreeLSTM(input_dim=dim, cell_dim=dim, vocab=snli.vocab)

# s1_lstm = DependencyNetwork(dim=dim, vocab=snli.vocab)
# s2_lstm = DependencyNetwork(dim=dim, vocab=snli.vocab)

classifier = MultiLayerPerceptron(di=2*dim, dh=dim, do=3)

def average(array, acc, span):
    for _ in range(round(len(array) / span) + 1):
        denom = len(array[:span])
        if denom > 0:
            val = sum(array[:span]) / span
            acc.append(val)
            array = array[span:]
    return acc

def compute_accuracy(data):
    n_correct = 0
    n_total = len(data)
    batchsize = 100
    label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
    for sample in data:
        s1_lstm.forward_pass(sample.sentence1)
        s2_lstm.forward_pass(sample.sentence2)

        s1 = s1_lstm.get_root_embedding()
        s2 = s2_lstm.get_root_embedding()

        xs = np.concatenate((s1, s2))
        ys = snli.binarize([sample.label])

        prediction = classifier.predict(xs)
        pred = label_dict[prediction[0]]
        if pred == sample.label:
            n_correct += 1

    print('Accuracy: ', n_correct / n_total)


def train(iters, batchsize, rate):
    for _ in range(iters):
        batch = random.sample(dev_data, batchsize)
        print(_)

        if _ == 25:
            rate = rate / 2.0
        if _ == 35:
            rate = rate / 2.0
        if _ == 43:
            rate = rate / 2.0

        for sample in batch:
            s1_lstm.forward_pass(sample.sentence1)
            s2_lstm.forward_pass(sample.sentence2)

            s1 = s1_lstm.get_root_embedding()
            s2 = s2_lstm.get_root_embedding()

            xs = np.concatenate((s1, s2))
            ys = snli.binarize([sample.label])

            classifier.train(xs, ys, rate=rate)

            emb1_grad = classifier.yi_grad[:dim]
            emb2_grad = classifier.yi_grad[dim:]

            s1_lstm.backward_pass(emb1_grad, rate=rate)
            s2_lstm.backward_pass(emb2_grad, rate=rate)
        

start_time = time.time()

compute_accuracy(dev_data)
train(iters, batchsize, rate)
compute_accuracy(dev_data)

print('Total runtime: ', time.time() - start_time)

averaged_costs = average(classifier.costs, [], batchsize)
classifier.costs = averaged_costs

plt.figure(figsize=(10,10))
plt.plot(range(len(classifier.costs)), classifier.costs)
plt.show()