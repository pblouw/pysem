import pickle
import spacy

import numpy as np

from collections import defaultdict, Counter
from spacy_utils import TokenWrapper

parser = spacy.load('en')


class Model(object):
    """
    """
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

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


def normalize(v):
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)


def most_common_dep(data):
    depcount = Counter()
    for sample in data:
        s1 = sample[0][0]
        s2 = sample[0][1]
        parse1 = parser(s1)
        parse2 = parser(s2)

        for token in parse1:
            if token.dep_ != 'ROOT':
                depcount[token.dep_] += 1

        for token in parse2:
            if token.dep_ != 'ROOT':
                depcount[token.dep_] += 1

    return depcount.most_common(1)[0][0]


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
        self.w1 = np.random.random((temp, di+1)) * eps * 2 - eps
        self.w2 = np.random.random((do, dh+1)) * eps * 2 - eps
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
            yh_grad = np.dot(self.w2.T, yo_grad) * self.tanh_grad(self.yh)
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
        self.parser = parser
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

    def initialize_weights(self, eps=0.2):
        self.weights = defaultdict(zeros(self.dim))
        self.vectors = {word: np.random.random((self.dim, 1)) *
                        eps * 2 - eps for word in self.vocab}
        for dep in self.depset:
            self.weights[dep] = self.gaussian_id(self.dim)

    def reset_comp_graph(self):
        for token in self.tree:
            token.computed = False

    def reset_embeddings(self):
        for token in self.tree:
            token._embedding = None

    def reset_gradients(self):
        for token in self.tree:
            token._gradient = None

    def reset_all(self):
        self.reset_embeddings()
        self.reset_comp_graph()
        self.reset_gradients()

    def clip_gradient(self, token):
        if np.linalg.norm(token.gradient) > 5:
            token.gradient = (token.gradient /
                              np.linalg.norm(token.gradient)) * 5

    def update_embeddings(self):
        for token in self.tree:
            try:
                self.vectors[token.lower_] += -self.rate * token.gradient
            except KeyError:
                pass

    def compute_gradients(self):
        for token in self.tree:
            if not self.has_children(token):
                continue

            if token.gradient is not None:
                self.clip_gradient(token)

                children = self.get_children(token)

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
            self.update_embeddings()
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
        self.rate = rate
        self.set_root_gradient(error_grad)
        self.compute_gradients()

        for dep in self.wgrads:
            self.weights[dep] += -rate * self.wgrads[dep]

        self.wgrads = defaultdict(zeros(self.dim))
        self.reset_comp_graph()
        self.reset_embeddings()
        self.reset_gradients()

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

    def embed(self, token, children=False):
        try:
            emb = np.copy(self.vectors[token.lower_])
        except KeyError:
            emb = np.zeros(self.dim).reshape((self.dim, 1))

        if children:
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
