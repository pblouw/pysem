import pickle
import spacy
import string

import numpy as np

from collections import defaultdict
from pysem.utils.spacy import TokenWrapper
from pysem.utils.vsa import normalize

parser = spacy.load('en')
punc_translator = str.maketrans({key: None for key in string.punctuation})


def square_zeros(dim):
    '''Returns a function that produces a square array of zeros when called.
    Used to initialize defaultdicts that default to a numpy array of zeros.'''
    def func():
        return np.zeros((dim, dim))
    return func


def flat_zeros(dim):
    '''Returns a function that produces a flat array of zeros when called.
    Used to initialize defaultdicts that default to a numpy array of zeros.'''
    def func():
        return np.zeros((dim, 1))
    return func


class RecursiveModel(object):
    """A base class for networks that use recursive applications of one or
    more set of weights to model sequential data. Recurrent networks model
    sequences by recursively applying weights in a linear chain, while
    dependency networks model sequences by recursively applying weights
    using tree structures.
    """
    @staticmethod
    def tanh(x):
        '''Apply the tanh nonlinearity to an input vector.'''
        return np.tanh(x)

    @staticmethod
    def tanh_grad(x):
        '''Compute tanh gradient with respect to an input vector.'''
        return 1.0 - x * x

    @staticmethod
    def softmax(x):
        '''Compute a softmax distribution over an input vector.'''
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def gaussian_id(dim):
        '''Returns an identity matrix with gaussian noise added.'''
        identity = np.eye(dim)
        gaussian = np.random.normal(loc=0, scale=0.05, size=(dim, dim))
        return identity + gaussian

    @staticmethod
    def random_weights(dim):
        '''Returns matrix of values sampled from +- 1/root(dim) interval.'''
        eps = 1.0 / np.sqrt(dim)
        weights = np.random.random((dim, dim)) * 2 * eps - eps
        return weights

    @staticmethod
    def random_vector(dim):
        '''Returns a random vector from the unit sphere.'''
        scale = 1 / np.sqrt(dim)
        vector = np.random.normal(loc=0, scale=scale, size=(dim, 1))
        return vector

    def pretrained_vecs(self, path):
        '''Load pretrained word embeddings for initialization.'''
        self.vectors = {}
        with open(path, 'rb') as pfile:
            pretrained = pickle.load(pfile)

        for word in self.vocab:
            try:
                self.vectors[word] = pretrained[word].reshape(self.dim, 1)
            except KeyError:
                scale = 1 / np.sqrt(self.dim)
                randvec = np.random.normal(0, scale=scale, size=(self.dim, 1))
                self.vectors[word] = normalize(randvec)

    def random_vecs(self):
        '''Use random word embeddings for initialization.'''
        scale = 1 / np.sqrt(self.dim)
        self.vectors = {word: normalize(np.random.normal(loc=0, scale=scale,
                        size=(self.dim, 1))) for word in self.vocab}


class DependencyNetwork(RecursiveModel):
    """A plain recurrent network that computes a hidden state given an input
    and the previous hidden state. The computed hidden state therefore depends
    on both the current input and the entire history of the input sequence up
    to this point. This implementation is designed to compress a sequence into
    a single hidden representation rather than make a prediction for each item
    in the input sequence (i.e. there are no hidden-to-output weights.)

    Parameters:
    ----------
    dim : int
        The dimensionality of the hidden state representation.
    vocab : list of strings
        The vocabulary of possible input items.
    eps : float, optional
        The scaling factor on random weight initialization.

    Attributes:
    -----------
    dim : int
        The dimensionality of the hidden state representation.
    vocab : list of strings
        The vocabulary of possible input items.
    parser : callable
        The parser used to produce a dependency tree from an input sentence.
    weights : defaultdict
        Matches each known dependency with the corresponding weight matrix.
    wgrads : defaultdict
        Matches each known dependency with the corresponding weight gradient.
    vectors : dict
        Matches each vocabulary item with a vector embedding that is learned
        over the course of training the network.
    tree : list
        A list of the nodes that make the dependency tree for an input
        sentence. Only computed when forward_pass is called on a sentence.
    """
    deps = ['compound', 'punct', 'nsubj', 'ROOT', 'det', 'attr', 'cc',
            'npadvmod', 'appos', 'prep', 'pobj', 'amod', 'advmod', 'acl',
            'nsubjpass', 'auxpass', 'agent', 'advcl', 'aux', 'xcomp', 'nmod',
            'dobj', 'relcl', 'nummod', 'mark', 'pcomp', 'conj', 'poss',
            'ccomp', 'oprd', 'acomp', 'neg', 'parataxis', 'dep', 'expl',
            'preconj', 'case', 'dative', 'prt', 'quantmod', 'meta', 'intj',
            'csubj', 'predet', 'csubjpass']

    def __init__(self, dim, vocab, eps=0.3, pretrained=False):
        self.dim = dim
        self.vocab = sorted(list(vocab))
        self.parser = parser
        self.biases = defaultdict(flat_zeros(self.dim))
        self.bgrads = defaultdict(flat_zeros(self.dim))
        self.weights = defaultdict(square_zeros(self.dim))
        self.wgrads = defaultdict(square_zeros(self.dim))
        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

        for dep in self.deps:
            self.weights[dep] = self.random_weights(self.dim)
            self.biases[dep] = np.zeros((self.dim, 1))

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
                    self.bgrads[child.dep_] += node.gradient
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
            emb += self.biases[child.dep_]

        node.embedding = self.tanh(emb)
        node.computed = True

    def update_word_embeddings(self):
        '''Use node gradients to update the word embeddings at each node.'''
        for node in self.tree:
            try:
                self.vectors[node.lower_] -= self.rate * node.gradient
            except KeyError:
                pass

    def update_weights(self):
        '''Use weight gradients to update the weights for each dependency,'''
        for dep in self.wgrads:
            depcount = len([True for node in self.tree if node.dep_ == dep])
            self.weights[dep] -= self.rate * self.wgrads[dep] / depcount
            self.biases[dep] -= self.rate * self.bgrads[dep] / depcount

    def forward_pass(self, sentence):
        '''Compute activations for every node in the computational graph
        generated from a dependency parse of the provided sentence.'''
        self.tree = [TokenWrapper(token) for token in self.parser(sentence)]
        self.compute_embeddings()
        self.reset_comp_graph()

    def backward_pass(self, error_grad, rate=0.35):
        '''Compute gradients for every weight matrix and input word vector
        used when computing activations in accordance with the comp graph.'''
        self.wgrads = defaultdict(square_zeros(self.dim))
        self.bgrads = defaultdict(flat_zeros(self.dim))
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


class RecurrentNetwork(RecursiveModel):
    """A plain recurrent network that computes a hidden state given an input
    and the previous hidden state. The computed hidden state therefore depends
    on both the current input and the entire history of the input sequence up
    to this point. This implementation is designed to compress a sequence into
    a single hidden representation rather than make a prediction for each item
    in the input sequence (i.e. there are no hidden-to-output weights.)

    Parameters:
    ----------
    dim : int
        The dimensionality of the hidden state representation.
    vocab : list of strings
        The vocabulary of possible input items.
    eps : float, optional
        The scaling factor on random weight initialization.

    Attributes:
    -----------
    dim : int
        The dimensionality of the hidden state representation.
    vocab : list of strings
        The vocabulary of possible input items.
    weights : numpy.ndarray
        The hidden-to-hidden weight matrix.
    bias : numpy.ndarray
        The bias vector on the hidden state.
    vectors : dict
        Matches each vocabulary item with a vector embedding that is learned
        over the course of training the network.
    """
    def __init__(self, dim, vocab, eps=0.3, pretrained=False):
        self.dim = dim
        self.vocab = vocab
        self.weights = self.random_weights(dim)
        self.xs, self.hs = {}, {}
        self.bias = np.zeros((dim, 1))
        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

    def clip_gradient(self, gradient, clipval=5):
        '''Clip a large gradient so that its norm is equal to clipval.'''
        norm = np.linalg.norm(gradient)
        if norm > clipval:
            gradient = (gradient / norm) * 5

        return gradient

    def compute_embeddings(self):
        '''Compute network hidden states for each item in the sequence.'''
        self.hs[-1] = np.zeros((self.dim, 1))
        for i in range(len(self.sequence)):
            self.xs[i] = self.vectors[self.sequence[i]]
            self.hs[i] = np.dot(self.weights, self.hs[i-1]) + self.xs[i]
            self.hs[i] = np.tanh(self.hs[i] + self.bias)

    def forward_pass(self, sentence):
        '''Converts input sentence into sequence and compute hidden states.'''
        self.sequence = [s.lower() for s in sentence.split()]
        self.sequence = [s.translate(punc_translator) for s in self.sequence]
        self.sequence = [s for s in self.sequence if s in self.vocab]
        self.compute_embeddings()

    def backward_pass(self, error_grad, rate=0.1):
        '''Compute gradients for hidden-to-hidden weight matrix and input word
        vectors before performing weight updates.'''
        error_grad = error_grad * self.tanh_grad(self.get_root_embedding())
        dh = error_grad
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        dh_next = np.zeros_like(self.hs[0])

        for i in reversed(range(len(self.sequence))):
            if i < len(self.sequence) - 1:
                dh = np.dot(self.weights.T, dh_next)
                dh = dh * self.tanh_grad(self.hs[i])

            dw += np.dot(dh_next, self.hs[i].T)
            db += dh
            dh_next = dh

            self.vectors[self.sequence[i]] -= rate * dh

        self.dw = self.clip_gradient(dw)
        self.db = self.clip_gradient(db)

        self.weights -= rate * self.dw
        self.bias -= rate * self.db

    def get_root_embedding(self):
        '''Returns the embedding for the final/root node in the sequence.'''
        return self.hs[len(self.sequence)-1]
