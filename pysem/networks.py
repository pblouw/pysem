import pickle
import spacy
import string

import numpy as np

from collections import defaultdict
from pysem.utils.spacy import TokenWrapper

parser = spacy.load('en')
punc_translator = str.maketrans({key: None for key in string.punctuation})


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


def square_zeros(dim):
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
        self.w1 = np.random.random((dh, di+1)) * eps * 2 - eps
        self.w2 = np.random.random((do, dh+1)) * eps * 2 - eps
        self.costs = []
        self.bsize = 1

    @property
    def label_dict(self):
        return self._label_dict

    @label_dict.setter
    def label_dict(self, l_dict):
        if not isinstance(l_dict, dict):
            raise TypeError('Label Dict must be a dictionary')
        else:
            self._label_dict = l_dict

    @property
    def nl(self):
        return self._nl

    @nl.setter
    def nl(self, act_func):
        self._nl = act_func

    @property
    def nl_grad(self):
        return self._nl_grad

    @nl_grad.setter
    def nl_grad(self, grad_func):
        self._nl_grad = grad_func

    def get_activations(self, x):
        self.yh = self.nl(np.dot(self.w1, x))
        self.yh = np.vstack((np.ones(self.bsize), self.yh))
        self.yo = self.softmax(np.dot(self.w2, self.yh))

    def train(self, xs, ys, iters, bsize=1, rate=0.35):
        self.bsize = bsize
        xs = np.reshape(xs, (len(xs), 1))

        for _ in range(iters):
            self.ys = ys
            # Compute activations
            self.get_activations(xs)

            # Compute gradients
            yo_grad = self.yo-self.ys
            yh_grad = np.dot(self.w2.T, yo_grad) * self.nl_grad(self.yh)

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
        return np.sum(-np.log(self.yo) * self.ys) / float(self.bsize)

    def predict(self, x):
        x = np.reshape(x, (len(x), 1))
        self.get_activations(x)
        return self.label_dict[int(np.argmax(self.yo, axis=0))]


class RNN(Model):
    def __init__(self, vocab, dim, eps=0.05):
        self.dim = dim
        # Randomly initialize the three weight matrices
        self.wx = np.random.random((dim, len(vocab)))*eps*2-eps
        self.wh = np.random.random((dim, dim))*eps*2-eps
        self.xs, self.hs, self.ys = {}, {}, {}
        self.bh = np.zeros(dim)
        self.vocab = vocab

        self.wrd_to_ind = {word: idx for idx, word in enumerate(vocab)}
        self.ind_to_wrd = {idx: word for idx, word in enumerate(vocab)}

    def get_onehot(self, ind):
        onehot = np.zeros(len(self.vocab))
        onehot[ind] = 1
        return onehot

    def set_root_gradient(self, grad):
        self.root_grad = grad

    def get_activities(self, seq_in):
        self.hs[-1] = np.zeros(len(self.wh))
        for t in range(len(seq_in)):
            self.xs[t] = self.get_onehot(seq_in[t])
            self.hs[t] = np.tanh(np.dot(self.wx, self.xs[t]) +
                                 np.dot(self.wh, self.hs[t-1])+self.bh)

            self.ys[t] = np.tanh(self.hs[t])

    def forward_pass(self, sequence):
        self.sequence = [s.lower() for s in sequence.split() if
                         s.lower() in self.vocab]
        self.sequence = [s.translate(punc_translator) for s in self.sequence]
        self.sequence = [s for s in self.sequence if s in self.vocab]
        print([s in self.vocab for s in self.sequence])
        print(self.sequence)
        xs = np.array([self.wrd_to_ind[word] for word in self.sequence])
        self.get_activities(xs)

    def backward_pass(self, error_grad, rate=0.3):
        self.set_root_gradient(error_grad)

        wx_grad = np.zeros_like(self.wx)
        wh_grad = np.zeros_like(self.wh)
        bh_grad = np.zeros_like(self.bh)

        h_grads = {}
        h_grads[len(self.sequence)] = np.zeros(self.dim)

        for _ in reversed(range(len(self.sequence))):
            if _ == len(self.sequence) - 1:
                h_grads[_] = self.root_grad.flatten() * (1 - self.ys[_]**2)
            else:
                h_grads[_] = np.dot(self.wh.T, h_grads[_+1])
                h_grads[_] = h_grads[_] * (1 - self.hs[_]**2)
                wx_grad += np.outer(h_grads[_], self.xs[_])
                wh_grad += np.outer(h_grads[_+1], self.hs[_])
                bh_grad += h_grads[_]

        grads = [wx_grad, wh_grad, bh_grad]

        # Clip gradients to avoid explosions
        for _ in range(len(grads)):
            if np.linalg.norm(grads[_]) > 5:
                grads[_] = 5 * grads[_] / np.linalg.norm(grads[_])

        wx_grad = grads[0]
        wh_grad = grads[1]
        bh_grad = grads[2]

        self.wx += -rate * wx_grad
        self.wh += -rate * wh_grad
        self.bh += -rate * bh_grad

    def get_sentence_embedding(self):
        print(len(self.ys))
        return self.ys[len(self.sequence)-1]


class DependencyNetwork(Model):

    deps = ['compound', 'punct', 'nsubj', 'ROOT', 'det', 'attr', 'cc',
            'npadvmod', 'appos', 'prep', 'pobj', 'amod', 'advmod', 'acl',
            'nsubjpass', 'auxpass', 'agent', 'advcl', 'aux', 'xcomp', 'nmod',
            'dobj', 'relcl', 'nummod', 'mark', 'pcomp', 'conj', 'poss',
            'ccomp', 'oprd', 'acomp', 'neg', 'parataxis', 'dep', 'expl',
            'preconj', 'case', 'dative', 'prt', 'quantmod', 'meta', 'intj',
            'csubj', 'predet', 'csubjpass']

    def __init__(self, dim, vocab, eps=0.2):
        self.dim = dim
        self.vocab = vocab
        self.indices = {wrd: idx for idx, wrd in enumerate(self.vocab)}
        self.parser = parser
        self.weights = defaultdict(square_zeros(self.dim))
        self.wgrads = defaultdict(square_zeros(self.dim))

        self.vectors = {word: np.random.random((self.dim, 1)) *
                        eps * 2 - eps for word in self.vocab}

        for dep in self.deps:
            self.weights[dep] = self.gaussian_id(self.dim)

    def one_hot(self, token):
        '''Converts a spacy token into the correct onehot encoding.'''
        zeros = np.zeros(len(self.vocab))
        try:
            index = self.indices[token.lower_]
            zeros[index] = 1
        except KeyError:
            pass
        return zeros.reshape((len(zeros), 1))

    @staticmethod
    def gaussian_id(dim):
        '''Returns an identity matrix with gaussian noise added.'''
        identity = np.eye(dim)
        gaussian = np.random.normal(loc=0, scale=0.05, size=(dim, dim))
        return identity + gaussian

    def load_vecs(self, path):
        '''Load pretrained word embeddings for initialization.'''
        with open(path, 'rb') as pfile:
            self.vectors = pickle.load(pfile)

    def reset_comp_graph(self):
        '''Flag all nodes in the graph as being uncomputed.'''
        for node in self.tree:
            node.computed = False

    def clip_gradient(self, node, clipval=5):
        '''Clip a large gradient so that its norm is equal to clipval.'''
        norm = np.linalg.norm(node.gradient)
        if norm > clipval:
            node.gradient = (node.gradient / norm) * 5

    def compute_gradients(self):
        '''Compute gradients for every weight matrix and embedding by
        recursively computing gradients for embeddings and weight matrices
        whose parents have been computed. Recursion terminates when every
        embedding and weight matrix has a gradient.'''
        for node in self.tree:
            if not self.has_children(node):
                continue

            if node.computed:
                self.clip_gradient(node)
                children = self.get_children(node)

                for child in children:
                    if child.computed:
                        continue

                    wgrad = np.outer(node.gradient, child.embedding)
                    cgrad = np.dot(self.weights[child.dep_].T, node.gradient)

                    nlgrad = self.tanh_grad(child.embedding)
                    nlgrad = nlgrad.reshape((len(nlgrad), 1))

                    self.wgrads[child.dep_] += wgrad
                    child.gradient = cgrad * nlgrad
                    child.computed = True

        if all([node.computed for node in self.tree]):
            return
        else:
            self.compute_gradients()

    def compute_embeddings(self):
        '''Computes embeddings for all nodes in the graph by recursively
        computing the embeddings for nodes whose children have all been
        computed. Recursion terminates when every node has an embedding.'''
        for node in self.tree:
            if not node.computed:
                children = self.get_children(node)
                children_computed = [c.computed for c in children]

                if all(children_computed):
                    self.embed_node(node, children)

        nodes_computed = [node.computed for node in self.tree]
        if all(nodes_computed):
            return
        else:
            self.compute_embeddings()

    def embed_node(self, node, children):
        '''Computes the vector embedding for a node from the vector embeddings
        of its children. In the case of leaf nodes with no children, the
        vector for the word corresponding to the leaf node is used as the
        embedding.'''
        try:
            emb = np.copy(self.vectors[node.lower_])
        except KeyError:
            emb = np.zeros(self.dim).reshape((self.dim, 1))

        for child in children:
            emb += np.dot(self.weights[child.dep_], child.embedding)

        node.embedding = self.tanh(emb)
        node.computed = True

    def update_word_embeddings(self):
        '''Use node gradients to update the word embeddings at each node.'''
        for node in self.tree:
            try:
                self.vectors[node.lower_] += -self.rate * node.gradient
            except KeyError:
                pass

    def update_weights(self):
        '''Use weight gradients to update the weights for each dependency,'''
        for dep in self.wgrads:
            self.weights[dep] += -self.rate * self.wgrads[dep]

        self.wgrads = defaultdict(square_zeros(self.dim))

    def forward_pass(self, sentence):
        '''Compute activations for every node in the computational graph
        generated from a dependency parse of the provided sentence.'''
        self.tree = [TokenWrapper(token) for token in self.parser(sentence)]
        self.compute_embeddings()
        self.reset_comp_graph()

    def backward_pass(self, error_grad, rate=0.35):
        '''Compute gradients for every weight matrix and input word vector
        used when computing activations in accordance with the comp graph.'''
        self._set_root_gradient(error_grad)
        self.rate = rate

        self.compute_gradients()
        self.update_weights()
        self.update_word_embeddings()

    def get_children(self, node):
        '''Returns all nodes that are children of the provided node.'''
        children = []
        for other_node in self.tree:
            if other_node.idx in [child.idx for child in node.children]:
                children.append(other_node)

        return children

    def has_children(self, node):
        '''Check if node has children, return False for leaf nodes.'''
        return bool(node.children)

    def get_root_embedding(self):
        '''Returns the embedding for the root node in the tree.'''
        for node in self.tree:
            if node.head.idx == node.idx:
                return node.embedding

    def _set_root_gradient(self, grad):
        '''Set the error gradient on the root node in the comp graph.'''
        for node in self.tree:
            if node.head.idx == node.idx:
                node.gradient = grad
                node.computed = True
