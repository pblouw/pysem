import random
import platform
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from handlers import SNLI


class Model(object):
    """
    """
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


class MLP(Model):
    """
    A three layer neural network for performing classification.
    Parameters:
    -----------
    di : int
        The dimensionality of the input vectors being classified.
    dh : int
        The dimensionality of the hidden layer of the network.
    do : int
        The dimensionality of the output vector that encodes a classification
        decision. (i.e. a probability distribution over labels)
    eps : float, optional
        Scaling factor on random weight initialization. By default, the
        weightsare chosen from a uniform distribution on the interval
        [-0.1, 0.1].
    """
    def __init__(self, di, dh, do, eps=0.1):
        temp = dh
        self.w1 = np.random.random((temp, di+1))*eps*2-eps
        # self.w1_b = np.random.random((temp, di+1))*eps*2-eps
        self.w2 = np.random.random((do, dh+1))*eps*2-eps
        self.costs = []

    def get_activations(self, x):
        # self.yh_a = self.sigmoid(np.dot(self.w1_a, x1))
        # self.yh_b = self.sigmoid(np.dot(self.w1_b, x2))
        self.yh = self.sigmoid(np.dot(self.w1, x))
        # self.yh = np.vstack((np.ones(self.bsize), self.yh_a, self.yh_b))
        self.yh = np.vstack((np.ones(self.bsize), self.yh))

        self.yo = self.softmax(np.dot(self.w2, self.yh))

    def train(self, snli, iters, bsize=200, rate=0.3):
        self.bsize = bsize

        BoW = CountVectorizer(binary=True)
        BoW.fit(snli.vocab)

        snli.extractor = snli.get_xy_pairs
        data = {d for d in snli.train_data if d[1] != '-'}
        print('Training set size: ', len(data))

        for _ in range(iters):

            if _ % 10000 == 0:
                if _ != 0:
                    print('Completed ', _, ' training iterations!')

            if _ == 0.6 * iters:
                rate = 0.15
                print('Dropped rate to ', rate)
            if _ == 0.8 * iters:
                rate = 0.075
                print('Dropped rate to ', rate)

            batch = random.sample(data, self.bsize)

            # Turn batches into arrays
            prems = [s[0][0] for s in batch]
            hyps = [s[0][1] for s in batch]
            targs = [s[1].encode('ascii') for s in batch]

            prem_bag = BoW.transform(prems).toarray().T
            prem_bag = np.vstack((np.ones(self.bsize), prem_bag))

            hyp_bag = BoW.transform(hyps).toarray().T
            # hyp_bag = np.vstack((np.ones(self.bsize), hyp_bag))

            inp_bag = np.vstack((prem_bag, hyp_bag))
            self.targ_bag = self.binarize(targs)

            # Compute activations
            self.get_activations(inp_bag)

            # Compute gradients
            yo_grad = self.yo-self.targ_bag
            yh_grad = np.dot(self.w2.T, yo_grad)*(self.yh*(1-self.yh))
            w2_grad = np.dot(yo_grad, self.yh.T) / self.bsize
            w1_grad = np.dot(yh_grad[1:, :], inp_bag.T) / self.bsize

            # w1a_grad = np.dot(yh_grad[1:201, :], prem_bag.T) / self.bsize
            # w1b_grad = np.dot(yh_grad[201:, :], hyp_bag.T) / self.bsize

            # Update weights
            self.w1 += -rate * w1_grad
            # self.w1_a += -rate * w1a_grad
            # self.w1_b += -rate * w1b_grad
            self.w2 += -rate * w2_grad

            # Log the cost of the current weights
            self.costs.append(self.get_cost())

    def get_cost(self):
        return np.sum(-np.log(self.yo) * self.targ_bag) / float(self.bsize)

    def predict(self, x):
        self.get_activations(x)
        return np.argmax(self.yo, axis=0)

    def get_accuracy(self, snli, dataset):

        snli.extractor = snli.get_xy_pairs

        def tally(data):
            # Turn batches into arrays
            self.bsize = len(data)

            prems = [s[0][0] for s in data]
            hyps = [s[0][1] for s in data]
            targs = [s[1].encode('ascii') for s in data]

            prem_bag = BoW.transform(prems).toarray().T
            prem_bag = np.vstack((np.ones(len(data)), prem_bag))

            hyp_bag = BoW.transform(hyps).toarray().T
            # hyp_bag = np.vstack((np.ones(self.bsize), hyp_bag))

            inp_bag = np.vstack((prem_bag, hyp_bag))

            targs = self.binarize(targs)

            correct = sum(np.equal(self.predict(inp_bag),
                          np.argmax(targs, axis=0)))

            return correct

        if dataset == 'dev':
            data = {d for d in snli.dev_data if d[1] != '-'}
            self.bsize = len(data)
            return tally(data) / float(self.bsize)

        if dataset == 'train':
            correct = 0
            bsize = 10000
            for _ in range(55):
                data = []
                for __ in range(bsize):
                    try:
                        while True:
                            x = next(snli.train_data)
                            if x[1] != '-':
                                break
                        data.append(x)
                    except:
                        break
                correct += tally(data)

            snli.extractor = snli.get_xy_pairs
            return correct / float(len({d for d in snli.train_data
                                   if d[1] != '-'}))

    @staticmethod
    def binarize(label_list):
        lookup = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 1}
        y_idx = [lookup[l] for l in label_list]
        x_idx = range(len(label_list))

        vals = np.zeros((3, len(label_list)))
        vals[y_idx, x_idx] = 1
        return vals

if platform.system() == 'Linux':
    snlipath = '/home/pblouw/corpora/snli_1.0/'
    wikipath = '/home/pblouw/corpora/wikipedia'
    cachepath = '/home/pblouw/cache/'
else:
    snlipath = '/Users/peterblouw/corpora/snli_1.0/'
    wikipath = '/Users/peterblouw/corpora/wikipedia'
    cachepath = '/Users/peterblouw/cache/'

snli = SNLI(snlipath)
snli.load_vocab()

BoW = CountVectorizer(binary=True)
BoW.fit(snli.vocab)

model = MLP(2*len(BoW.get_feature_names()), 400, 3)
print('Dev Set Accuracy Before: ', model.get_accuracy(snli, 'dev'))

model.train(snli, iters=25000, bsize=600)

print('Train Set Accuracy After: ', model.get_accuracy(snli, 'train'))
print('Dev Set Accuracy After: ', model.get_accuracy(snli, 'dev'))

plt.figure()
plt.plot(np.arange(len(model.costs)), model.costs)
plt.show()
