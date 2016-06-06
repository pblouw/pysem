import pickle
import spacy
import platform
import random

import numpy as np

from collections import defaultdict
from handlers import SNLI
from spacy_utils import TokenWrapper

if platform.system() == 'Linux':
    snlipath = '/home/pblouw/corpora/snli_1.0/'
    wikipath = '/home/pblouw/corpora/wikipedia'
    cachepath = '/home/pblouw/cache/'
else:
    snlipath = '/Users/peterblouw/corpora/snli_1.0/'
    wikipath = '/Users/peterblouw/corpora/wikipedia'
    cachepath = '/Users/peterblouw/cache/'


class Model(object):
    """
    """
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_grad(x):
        return 1.0 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


def zeros(dim):
    def func():
        return np.zeros((dim, dim))
    return func


class MLP(Model):
    """
    A multilayer perceptron performing classification.
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
        self.w2 = np.random.random((do, dh+1))*eps*2-eps
        self.costs = []
        self.bsize = 1
        self.node_to_label = {0: 'entailment', 1: 'neutral',
                              2: 'contradiction'}

    def get_activations(self, x):
        self.yh = self.tanh(np.dot(self.w1, x))
        self.yh = np.vstack((np.ones(self.bsize), self.yh))

        self.yo = self.softmax(np.dot(self.w2, self.yh))

    def train(self, xs, ys, iters, bsize=1, rate=0.35):
        self.bsize = bsize
        xs = np.reshape(xs, (len(xs), 1))

        for _ in range(iters):
            self.targ_bag = self.binarize([ys])

            # Compute activations
            self.get_activations(xs)

            # Compute gradients
            yo_grad = self.yo-self.targ_bag
            yh_grad = np.dot(self.w2.T, yo_grad)*self.tanh_grad(self.yh)
            w2_grad = np.dot(yo_grad, self.yh.T) / self.bsize
            w1_grad = np.dot(yh_grad[1:, :], xs.T) / self.bsize

            self.yi_grad = np.dot(self.w1.T, yh_grad[1:])
            self.yo_grad = yo_grad
            self.yh_grad = yh_grad
            # Update weights
            self.w1 += -rate * w1_grad
            self.w2 += -rate * w2_grad

            # Log the cost of the current weights
            self.costs.append(self.get_cost())

    def get_cost(self):
        return np.sum(-np.log(self.yo) * self.targ_bag) / float(self.bsize)

    def predict(self, x):
        x = np.reshape(x, (len(x), 1))
        self.get_activations(x)
        return self.node_to_label[int(np.argmax(self.yo, axis=0))]

    def get_accuracy(self, snli, dataset):

        snli.extractor = snli.get_xy_pairs

        def tally(data):
            # Turn batches into arrays
            self.bsize = len(data)

            # prems = [s[0][0] for s in data]
            # hyps = [s[0][1] for s in data]
            # targs = [s[1] for s in data]

            # prem_bag = BoW.transform(prems).toarray().T
            # prem_bag = np.vstack((np.ones(len(data)), prem_bag))

            # hyp_bag = BoW.transform(hyps).toarray().T
            # inp_bag = np.vstack((prem_bag, hyp_bag))

            # targs = self.binarize(targs)

            # correct = sum(np.equal(self.predict(inp_bag),
            #               np.argmax(targs, axis=0)))

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


class DependencyNetwork(Model):

    def __init__(self, embedding_dim, vocab):
        self.dim = embedding_dim
        self.vocab = sorted(vocab)
        self.indices = {wrd: idx for idx, wrd in enumerate(self.vocab)}
        self.parser = spacy.load('en')
        self.load_dependencies('dependencies.pickle')
        self.wgrads = defaultdict(zeros(self.dim))
        self.initialize_weights()

    def one_hot(self, token):
        zeros = np.zeros(len(self.vocab))
        try:
            index = self.indices[token.lower_]
            zeros[index] = 1
        except KeyError:
            pass
        return zeros.reshape((len(zeros), 1))

    @staticmethod
    def gaussian_id(dim):
        identity = np.eye(dim)
        gaussian = np.random.normal(loc=0, scale=0.05, size=(dim, dim))
        return identity + gaussian

    def load_dependencies(self, path):
        with open(path, 'rb') as pfile:
            self.depset = pickle.load(pfile)

    def initialize_weights(self, eps=0.1):
        self.weights = {'token': np.random.random((self.dim,
                        len(self.vocab)))*eps*2-eps}
        self.wgrads['token'] = np.zeros((self.dim, len(self.vocab)))
        for dep in self.depset:
            self.weights[dep] = self.gaussian_id(self.dim)

    def reset_comp_graph(self):
        for token in self.tree:
            token.computed = False

    def reset_embeddings(self):
        for token in self.tree:
            token._embedding = None

    def clip_gradient(self, token):
        if np.linalg.norm(token.gradient) > 5:
            token.gradient = (token.gradient /
                              np.linalg.norm(token.gradient)) * 5

    def update_word_embeddings(self):
        for token in self.tree:
            one_hot = self.one_hot(token)
            self.wgrads['token'] += np.outer(token.gradient, one_hot)

    def compute_gradients(self):
        for token in self.tree:
            if not self.has_children(token):
                continue

            if token.gradient is not None:
                self.clip_gradient(token)

                children = self.get_children(token)

                if np.isnan(token.gradient).any():
                    raise ValueError('NaN encountered')

                for child in children:
                    if child.gradient is not None:
                        continue

                    self.wgrads[child.dep_] += np.outer(token.gradient,
                                                        child.embedding)
                    child.gradient = np.dot(self.weights[child.dep_].T,
                                            token.gradient)
                    nl = self.tanh_grad(child.embedding)
                    nl = nl.reshape((len(nl), 1))

                    child.gradient = child.gradient * nl
                    child.computed = True

        grads_computed = [t.computed for t in self.tree]
        if all(grads_computed):
            self.update_word_embeddings()
            return
        else:
            self.compute_gradients()

    def forward_pass(self, sentence):
        self.sentence = sentence
        self.tree = [TokenWrapper(t) for t in self.parser(sentence)]
        self.compute_leaves()
        self.compute_nodes()
        self.reset_comp_graph()

    def backward_pass(self, error_grad, rate=0.35):
        self.set_root_gradient(error_grad)
        self.compute_gradients()

        for dep in self.wgrads:
            self.weights[dep] += -rate * self.wgrads[dep]

        self.wgrads = defaultdict(zeros(self.dim))
        self.wgrads['token'] = np.zeros((self.dim, len(self.vocab)))
        self.reset_comp_graph()
        self.reset_embeddings()

    def get_children(self, token):
        children = []
        for other_token in self.tree:
            if other_token.idx in [child.idx for child in token.children]:
                children.append(other_token)

        return children

    def has_children(self, token):
        if list(token.children) == list():
            return False
        else:
            return True

    def embed(self, token, children=list()):
        try:
            emb = np.dot(self.weights['token'], self.one_hot(token))
        except KeyError:
            emb = np.zeros(self.dim).reshape((self.dim, 1))

        for child in children:
            emb += np.dot(self.weights[child.dep_], child.embedding)

        token.embedding = self.tanh(emb)
        token.computed = True

    def compute_leaves(self):
        for token in self.tree:
            if not self.has_children(token):
                self.embed(token)

    def compute_nodes(self):
        for token in self.tree:
            if self.has_children(token) and not token.computed:
                children = self.get_children(token)
                children_computed = [child.computed for child in children]

                if all(children_computed):
                    self.embed(token, children)

        nodes_computed = [token.computed for token in self.tree]
        if all(nodes_computed):
            return
        else:
            self.compute_nodes()

    def set_root_gradient(self, grad):
        for token in self.tree:
            if token.head.idx == token.idx:
                token.gradient = grad
                token.computed = True

    def get_sentence_embedding(self):
        for token in self.tree:
            if token.head.idx == token.idx:
                return token.embedding


snli = SNLI(snlipath)
snli.extractor = snli.get_xy_pairs
snli.load_vocab('snli_words')

s1_depnet = DependencyNetwork(embedding_dim=200, vocab=snli.vocab)
s2_depnet = DependencyNetwork(embedding_dim=200, vocab=snli.vocab)

classifier = MLP(di=400, dh=400, do=3)

data = [d for d in snli.dev_data if d[1] != '-']
data = random.sample(data, 500)

iters = 1
rate = 0.1
counter = 0
for _ in range(iters):
    print('On training iteration ', _)
    # samples = random.sample(data, 5000)
    if _ == 0.6 * iters:
        rate = 0.05
        print('Dropped rate to ', rate)
    if _ == 0.8 * iters:
        rate = 0.025
        print('Dropped rate to ', rate)

    for sample in data:

        s1 = sample[0][0]
        s2 = sample[0][1]
        label = sample[1]

        s1_depnet.forward_pass(s1)
        s2_depnet.forward_pass(s2)

        bias = np.ones(1).reshape(1, 1)
        s1 = s1_depnet.get_sentence_embedding()
        s2 = s2_depnet.get_sentence_embedding()

        # print(bias.shape, s1.shape, s2.shape)

        xs = np.concatenate((bias, s1, s2))
        ys = label

        classifier.train(xs, ys, iters=1, rate=rate)
        s1_grad = classifier.yi_grad[1:201]
        s2_grad = classifier.yi_grad[201:]

        s1_depnet.backward_pass(s1_grad, rate=rate)
        s2_depnet.backward_pass(s2_grad, rate=rate)
    counter += 1
    # print(classifier.predict(xs))
    # print(label)

print('Calculating accuracies')
snli._reset_streams()
snli.extractor = snli.get_xy_pairs

# data = [d for d in snli.dev_data if d[1] != '-']
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
