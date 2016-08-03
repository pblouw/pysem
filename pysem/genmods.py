import spacy

import numpy as np

from collections import defaultdict
from pysem.utils.spacy import TokenWrapper
from pysem.networks import DependencyNetwork, square_zeros

parser = spacy.load('en')


class GenerativeDependencyNetwork(DependencyNetwork):
    """
    A model that generates predictions for the words occupying each node in
    the dependency tree corresponding to a supplied sentence.
    """
    def __init__(self, dim, vocab, depdict):
        self.dim = dim
        self.vocab = sorted(list(vocab))
        self.parser = parser
        self.weights = defaultdict(square_zeros(self.dim))
        self.wgrads = defaultdict(square_zeros(self.dim))
        self.depdict = depdict

        for dep in self.deps:
            self.weights[dep] = self.random_weights(self.dim)

        self.build_word_predictors(depdict)

    def build_word_predictors(self, depdict):
        '''Initialize random weights for predict words sorted by the dependencies
        they can occupy relative to a head word.'''
        self.w_weights = {}
        self.w_indices = {}
        self.i_indices = {}
        self.vocabs = {}

        eps = 1.0 / np.sqrt(self.dim)
        for dep in self.deps:
            n = len(depdict[dep]) if len(depdict[dep]) > 0 else 1
            scale = 2 * eps - eps
            self.w_weights[dep] = np.random.random((n, self.dim)) * scale

        for dep in depdict:
            words = sorted(depdict[dep])
            self.w_indices[dep] = {ind: wrd for ind, wrd in enumerate(words)}
            self.i_indices[dep] = {wrd: ind for ind, wrd in enumerate(words)}
            self.vocabs[dep] = words

    def forward_pass(self, sentence, root_embedding):
        '''Predict words at each node in the tree given a root embedding'''
        self.tree = [TokenWrapper(token) for token in self.parser(sentence)]
        self.compute_embeddings(root_embedding)
        self.reset_comp_graph()

    def backward_pass(self, rate=0.01):
        '''Compute gradients and update every weight matrix used to predict
        words at each node in the tree.'''
        self.wgrads = defaultdict(square_zeros(self.dim))
        self.rate = rate
        self.compute_gradients()
        self.update_weights()

    def compute_embeddings(self, root_embedding):
        '''Compute distributed representations at each node that can be used to
        predict the word occupying the node.'''
        for node in self.tree:
            if node.head.idx == node.idx:
                node.embedding = np.tanh(root_embedding)
                node.computed = True
                product = np.dot(self.w_weights[node.dep_], node.embedding)
                node.probs = self.softmax(product)
                idx = np.argmax(node.probs, axis=0)[0]
                node.pword = self.w_indices[node.dep_][idx]
                node.tvals = self.get_target_dist(node)

        for node in self.tree:
            parent = self.get_parent(node)
            if not node.computed and parent.computed:
                extract = np.dot(self.weights[node.dep_], parent.embedding)
                extract = np.tanh(extract)
                node.embedding = extract
                node.computed = True
                product = np.dot(self.w_weights[node.dep_], node.embedding)
                node.probs = self.softmax(product)
                idx = np.argmax(node.probs, axis=0)[0]
                node.pword = self.w_indices[node.dep_][idx]
                node.tvals = self.get_target_dist(node)

        computed = [node.computed for node in self.tree]
        if all(computed):
            return
        else:
            self.compute_embeddings(root_embedding)

    def compute_gradients(self):
        '''Compute gradients at each node by backpropogating while using info
        about the correct word at each node.'''
        for node in self.tree:
            children = self.get_children(node)
            children_computed = [c.computed for c in children]

            if not node.computed and all(children_computed):
                predgrad = node.probs - node.tvals
                ngrad = np.dot(self.w_weights[node.dep_].T, predgrad)
                wp_grad = np.dot(predgrad, node.embedding.T)

                self.w_weights[node.dep_] -= self.rate * wp_grad

                for child in children:
                    wgrad = np.dot(child.gradient, node.embedding.T)
                    ngrad += np.dot(self.weights[child.dep_].T, child.gradient)
                    self.wgrads[child.dep_] += wgrad

                node.gradient = self.tanh_grad(node.embedding) * ngrad
                self.clip_gradient(node)
                node.computed = True

        if all([n.computed for n in self.tree]):
            for node in self.tree:
                if node.head.idx == node.idx:
                    self.pass_grad = node.gradient
            self.update_weights()
            return
        else:
            self.compute_gradients()

    def get_target_dist(self, node):
        '''Compute a target softmax distribution given the correct word at
        a node.'''
        array = np.zeros_like(node.probs)
        idx = self.i_indices[node.dep_][node.lower_]
        array[idx] = 1
        return array

    def get_parent(self, node):
        '''Get the node that is the parent of the supplied node'''
        for other_node in self.tree:
            if other_node.idx == node.head.idx:
                return other_node

        raise ValueError('No parent found for the node')

    def update_weights(self):
        '''Use gradients to update the weights/biases for each dependency.'''
        for dep in self.wgrads:
            depcount = len([True for node in self.tree if node.dep_ == dep])
            self.weights[dep] -= self.rate * self.wgrads[dep] / depcount
