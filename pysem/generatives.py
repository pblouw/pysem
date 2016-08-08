import numpy as np

from collections import defaultdict
from pysem.utils.spacy import TokenWrapper
from pysem.networks import DependencyNetwork, square_zeros


class EmbeddingGenerator(DependencyNetwork):
    """
    A model that generates predictions for the words occupying each node in
    the dependency tree corresponding to a supplied sentence.
    """
    def __init__(self, dim, subvocabs):
        self.dim = dim
        self.weights = defaultdict(square_zeros(self.dim))
        self.wgrads = defaultdict(square_zeros(self.dim))
        self.subvocabs = subvocabs

        for dep in self.deps:
            self.weights[dep] = self.random_weights(self.dim)

        self.build_word_predictors()

    def build_word_predictors(self):
        '''Initialize random weights for predicted words sorted by the dependencies
        they can occupy relative to a head word.'''
        self.word_weights = {}
        self.idx_to_wrd = {}
        self.wrd_to_idx = {}

        eps = 1.0 / np.sqrt(self.dim)

        for dep in self.deps:
            n = len(self.subvocabs[dep]) if len(self.subvocabs[dep]) > 0 else 1
            scale = 2 * eps - eps
            self.word_weights[dep] = np.random.random((n, self.dim)) * scale

            words = self.subvocabs[dep]
            self.idx_to_wrd[dep] = {ind: wrd for ind, wrd in enumerate(words)}
            self.wrd_to_idx[dep] = {wrd: ind for ind, wrd in enumerate(words)}

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
                product = np.dot(self.word_weights[node.dep_], node.embedding)
                node.probs = self.softmax(product)
                idx = np.argmax(node.probs, axis=0)[0]
                node.pword = self.idx_to_wrd[node.dep_][idx]
                node.tvals = self.get_target_dist(node)

        for node in self.tree:
            parent = self.get_parent(node)
            if not node.computed and parent.computed:
                extract = np.dot(self.weights[node.dep_], parent.embedding)
                extract = np.tanh(extract)
                node.embedding = extract
                node.computed = True
                product = np.dot(self.word_weights[node.dep_], node.embedding)
                node.probs = self.softmax(product)
                idx = np.argmax(node.probs, axis=0)[0]
                node.pword = self.idx_to_wrd[node.dep_][idx]
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
                ngrad = np.dot(self.word_weights[node.dep_].T, predgrad)
                wp_grad = np.dot(predgrad, node.embedding.T)

                self.word_weights[node.dep_] -= self.rate * wp_grad

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
        idx = self.wrd_to_idx[node.dep_][node.lower_]
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


class TreeGenerator(DependencyNetwork):

    def __init__(self, dim, vocab, eps=0.1):
        self.dim = dim
        self.vocab = sorted(list(vocab))

        self.idx_to_wrd = {ind: wrd for ind, wrd in enumerate(self.vocab)}
        self.wrd_to_idx = {wrd: ind for ind, wrd in enumerate(self.vocab)}
        self.idx_to_dep = {ind: dep for ind, dep in enumerate(self.deps)}
        self.dep_to_idx = {dep: ind for ind, dep in enumerate(self.deps)}

        self.random_vecs()

        # add vector embeddings for each dep to embedding dict
        for dep in self.deps:
            self.vectors[dep] = self.random_vector(self.dim)

        # randomly initialize weight matrices
        scale = 2 * eps - eps
        self.why_w = np.random.random((len(self.vocab), dim)) * scale
        self.why_h = np.random.random((len(self.vocab), dim)) * scale
        self.why_d = np.random.random((len(self.deps), dim)) * scale

        self.wxh = defaultdict(square_zeros(self.dim))
        self.whh = self.random_weights(self.dim)

        for dep in self.deps:
            self.wxh[dep] = self.random_weights(self.dim)

    def compute_embeddings(self, node):
        # compute hidden state from input and previous hidden state
        hh_inp = np.dot(self.whh, self.p_emb)
        xh_inp = np.dot(self.wxh[node.dep_], self.x_emb)

        # set hidden state and predictions as node attributes
        node.embedding = np.tanh(hh_inp + xh_inp)
        node.input = self.x_emb

        node.py_w = self.softmax(np.dot(self.why_w, node.embedding))
        node.py_h = self.softmax(np.dot(self.why_h, node.embedding))
        node.py_d = self.softmax(np.dot(self.why_d, node.embedding))

        # set prediction targets as node attributes
        node.ty_w = self.get_word_target(node)
        node.ty_h = self.get_head_target(node)
        node.ty_d = self.get_dep_target(node)

        # set predicted words, heads, deps, as node attributes
        self.p_emb = node.embedding
        node.pw = self.idx_to_wrd[np.argmax(node.py_w, axis=0)[0]]
        node.ph = self.idx_to_wrd[np.argmax(node.py_h, axis=0)[0]]
        node.pd = self.idx_to_dep[np.argmax(node.py_d, axis=0)[0]]

        node.computed = True

    def forward_pass(self, embedding, target_sentence, train=True):
        self.tree = [TokenWrapper(n) for n in self.parser(target_sentence)]
        self.root_embedding = embedding
        self.sequence = []

        self.p_emb = np.zeros_like(embedding)
        self.x_emb = embedding

        for node in self.tree:
            if node.head.idx == node.idx:
                self.compute_embeddings(node)
                self.sequence.append(node)

        if len(self.sequence) == len(self.tree):
            return
        else:
            self.extend_sequence()

    def extend_sequence(self):
        for node in self.tree:
            if not node.computed:
                parent = self.get_parent(node)

                if parent in self.sequence and parent.computed:
                    ctx = [node.head.lower_, node.dep_, node.lower_]
                    self.x_emb = sum([np.copy(self.vectors[c]) for c in ctx
                                      if c in self.vectors])
                    self.compute_embeddings(node)
                    self.sequence.append(node)

    def backward_pass(self, rate=0.01):
        dwxh = defaultdict(square_zeros(self.dim))
        dwhh = np.zeros_like(self.whh)

        dwhy_w = np.zeros_like(self.why_w)
        dwhy_d = np.zeros_like(self.why_d)
        dwhy_h = np.zeros_like(self.why_h)

        dh_next = np.zeros_like(self.sequence[-1].embedding)

        for node in reversed(self.sequence):
            dh = np.zeros_like(self.sequence[-1].embedding)

            dy_h = node.py_h - node.ty_h
            dy_w = node.py_w - node.ty_w
            dy_d = node.py_d - node.ty_d

            dwhy_w += np.dot(dy_w, node.embedding.T)
            dwhy_h += np.dot(dy_h, node.embedding.T)
            dwhy_d += np.dot(dy_d, node.embedding.T)

            dh += np.dot(self.whh.T, dh_next)
            dh += np.dot(self.why_w.T, dy_w)
            dh += np.dot(self.why_h.T, dy_h)
            dh += np.dot(self.why_d.T, dy_d)
            dh = self.tanh_grad(node.embedding) * dh

            dwhh += np.dot(dh_next, node.embedding.T)
            dh_next = dh

            dwxh[node.dep_] += np.dot(dh, node.input.T)

        inp_grad = np.dot(self.wxh[node.dep_].T, dh)
        inp_grad = inp_grad * self.root_embedding
        self.pass_grad = inp_grad

        self.why_w -= rate * (dwhy_w / len(self.sequence))
        self.why_d -= rate * (dwhy_d / len(self.sequence))
        self.why_h -= rate * (dwhy_h / len(self.sequence))
        self.whh -= rate * (dwhh / len(self.sequence))

        for dep in dwxh:
            count = sum([1 for node in self.sequence if node.dep_ == dep])
            self.wxh[dep] -= rate * (dwxh[dep] / count)

    def get_word_target(self, node):
        array = np.zeros_like(node.py_w)
        idx = self.wrd_to_idx[node.lower_]
        array[idx] = 1
        return array

    def get_head_target(self, node):
        array = np.zeros_like(node.py_h)
        idx = self.wrd_to_idx[node.head.lower_]
        array[idx] = 1
        return array

    def get_dep_target(self, node):
        array = np.zeros_like(node.py_d)
        idx = self.dep_to_idx[node.dep_]
        array[idx] = 1
        return array

    def get_parent(self, node):
        '''Get the node that is the parent of the supplied node'''
        for other_node in self.tree:
            if other_node.idx == node.head.idx:
                return other_node

        raise ValueError('No parent found for the node')
