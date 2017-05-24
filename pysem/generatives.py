import pickle
import random
import numpy as np

from collections import defaultdict
from pysem.utils.spacy import TokenWrapper
from pysem.networks import DependencyNetwork, SquareZeros


class Node(object):
    """A dummy tree node for predicting tree structure."""
    def __init__(self, word):
        self.lower_ = word


class EmbeddingGenerator(DependencyNetwork):
    """A model that generates predictions for the words occupying each node in
    the dependency tree corresponding to a supplied sentence. The model takes
    a distributed representation that conditions the generation process as
    input, along with the sentence being predicted. The supplied sentence is
    used to generate a dependency tree that defines computational graph that
    is used to implement the generation process. Computing the states at each
    node in this graph then produces a word prediction at each node.

    Parameters:
    -----------
    dim : int
        The dimensionality of the hidden state representations.
    subvocabs : dict
        A dictionary that maps each dependency to a collection of words.

    Attributes:
    -----------
    dim : int
        The dimensionality of the hidden state representation.
    subvocabs : dict
        A dictionary that maps each dependency to a collection of words.
    parser : callable
        The parser used to produce a dependency tree from a provided sentence.
    d_weights : dict
        Matches each known dependency with a corresponding weight matrix.
    w_weights : dict
        Matches each known dependency with a corresponding weight matrix for
        predicting the word occupying the dependency in question.
    dgrads : defaultdict
        Matches each known dependency with the corresponding weight gradient.
    wgrads : defaultdict
        Matches each known dependency with a corresponding weight matrix for
        predicting the word occupying the dependency in question.
    tree : list
        A list of the nodes that make up the dependency tree for predicted
        sentence. Computed when forward_pass is called.
    """
    def __init__(self, dim, subvocabs=None, vectors=None):
        self.dim = dim
        self.subvocabs = subvocabs

        self.d_weights = {}
        self.w_weights = {}
        self.idx_to_wrd = {}
        self.wrd_to_idx = {}

        for dep in self.deps:
            self.d_weights[dep] = self.gaussian_id(self.dim)

        if subvocabs is not None:
            self.pretrained_vecs(vectors) if vectors else self.random_vecs()

    def load(self, filename):
        '''Load model parameters from a pickle file.'''
        with open(filename, 'rb') as pfile:
            params = pickle.load(pfile)

        self.dim = params['dim']
        self.subvocabs = params['subvocabs']
        self.d_weights = params['d_weights']
        self.w_weights = params['w_weights']
        self.idx_to_wrd = params['idx_to_wrd']
        self.wrd_to_idx = params['wrd_to_idx']

    def save(self, filename):
        '''Save model parameters to a pickle file.'''
        params = {'d_weights': self.d_weights, 'w_weights': self.w_weights,
                  'subvocabs': self.subvocabs, 'idx_to_wrd': self.idx_to_wrd,
                  'wrd_to_idx': self.wrd_to_idx, 'dim': self.dim}

        with open(filename, 'wb') as pfile:
            pickle.dump(params, pfile)

    def pretrained_vecs(self, vectors):
        with open(vectors, 'rb') as pfile:
            pretrained = pickle.load(pfile)

        eps = 1.0 / np.sqrt(self.dim)
        for dep in self.deps:
            vsize = len(self.subvocabs[dep])
            vsize = vsize if vsize > 0 else 1
            self.w_weights[dep] = np.zeros((vsize, self.dim))

            for idx, word in enumerate(self.subvocabs[dep]):
                try:
                    self.w_weights[dep][idx, :] = np.copy(pretrained[word])
                except KeyError:
                    vec = np.random.normal(0, scale=eps, size=self.dim)
                    self.w_weights[dep][idx, :] = vec

            vocab = self.subvocabs[dep]
            self.idx_to_wrd[dep] = {ind: wrd for ind, wrd in enumerate(vocab)}
            self.wrd_to_idx[dep] = {wrd: ind for ind, wrd in enumerate(vocab)}

    def random_vecs(self):
        eps = 1.0 / np.sqrt(self.dim)

        for dep in self.deps:
            vsize = len(self.subvocabs[dep])
            vsize = vsize if vsize > 0 else 1
            self.w_weights[dep] = np.random.random((vsize, self.dim)) * eps

            vocab = self.subvocabs[dep]
            self.idx_to_wrd[dep] = {ind: wrd for ind, wrd in enumerate(vocab)}
            self.wrd_to_idx[dep] = {wrd: ind for ind, wrd in enumerate(vocab)}

    def forward_pass(self, sentence, root_embedding):
        '''Predict words at each node in the tree given a root embedding'''
        self.tree = [TokenWrapper(token) for token in self.parser(sentence)]
        self.compute_embeddings(root_embedding)

    def backward_pass(self, rate=0.01):
        '''Compute gradients and update every weight matrix used to predict
        words at each node in the tree.'''
        self.reset_comp_graph()
        self.dgrads = defaultdict(SquareZeros(self.dim))
        self.wgrads = {d: np.zeros_like(self.w_weights[d]) for d in self.deps}
        self.rate = rate
        self.compute_gradients()
        self.update_weights()

    def compute_embeddings(self, root_embedding):
        '''Compute distributed representations at each node that can be used to
        predict the word occupying the node.'''
        for node in self.tree:
            if node.head.idx == node.idx:
                node.embedding = root_embedding
                node.computed = True
                product = np.dot(self.w_weights[node.dep_], node.embedding)
                node.probs = self.softmax(product)
                idx = np.argmax(node.probs, axis=0)[0]
                node.pword = self.idx_to_wrd[node.dep_][idx]
                node.tvals = self.get_target_dist(node)

        for node in self.tree:
            parent = self.get_parent(node)
            if not node.computed and parent.computed:
                extract = np.dot(self.d_weights[node.dep_], parent.embedding)
                extract = np.tanh(extract)
                node.embedding = extract
                node.computed = True
                product = np.dot(self.w_weights[node.dep_], node.embedding)
                node.probs = self.softmax(product)
                idx = np.argmax(node.probs, axis=0)[0]

                try:
                    node.pword = self.idx_to_wrd[node.dep_][idx]
                except KeyError:
                    node.pword = 'none'

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
                dp = node.probs - node.tvals
                dn = np.dot(self.w_weights[node.dep_].T, dp)
                dw = np.dot(dp, node.embedding.T)

                self.wgrads[node.dep_] += dw

                for child in children:
                    dd = np.dot(child.gradient, node.embedding.T)
                    dn += np.dot(self.d_weights[child.dep_].T, child.gradient)
                    self.dgrads[child.dep_] += dd

                if node.head.idx == node.idx:
                    node.gradient = dn
                else:
                    node.gradient = self.tanh_grad(node.embedding) * dn

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
        try:
            idx = self.wrd_to_idx[node.dep_][node.lower_]
            array[idx] = 1
        except KeyError:
            pass

        return array

    def get_parent(self, node):
        '''Get the node that is the parent of the supplied node'''
        for other_node in self.tree:
            if other_node.idx == node.head.idx:
                return other_node

        raise ValueError('No parent found for the node')

    def get_cost(self, sentence, embedding):
        '''Evaluate cost of the current parameters for gradient checking.'''
        self.reset_comp_graph()
        self.forward_pass(sentence, embedding)

        cost = 0
        for node in self.tree:
            cost += np.sum(-np.log(node.probs) * node.tvals)

        return cost

    def update_weights(self):
        '''Use gradients to update the weights/biases for each dependency.'''
        for dep in self.dgrads:
            count = sum([1 for node in self.tree if node.dep_ == dep])
            self.d_weights[dep] -= self.rate * self.dgrads[dep] / count
            self.w_weights[dep] -= self.rate * self.wgrads[dep] / count


class EncoderDecoder(object):

    def __init__(self, encoder=None, decoder=None, data=None):
        self.encoder = encoder
        self.decoder = decoder
        self.data = data

    def load(self, enc_filename, dec_filename):
        '''Load model parameters from a pickle file.'''
        self.encoder = DependencyNetwork(dim=1, vocab=['a'])
        self.encoder.load(enc_filename)

        self.decoder = EmbeddingGenerator(dim=1, subvocabs=None)
        self.decoder.load(dec_filename)

    def save(self, enc_filename, dec_filename):
        self.encoder.save(enc_filename)
        self.decoder.save(dec_filename)

    def train(self, iters, rate, schedule=25):
        self.rate = rate

        for i in range(iters):
            print('On iteration ', i)
            if i % schedule == 0 and i != 0:
                self.rate = self.rate / 2.0
                print('Learning rate annealed to ', self.rate)

            for sample in self.data:
                self.encoder.forward_pass(sample.sentence1)
                self.decoder.forward_pass(sample.sentence2,
                                          self.encoder.get_root_embedding())

                self.decoder.backward_pass(rate=rate)
                self.encoder.backward_pass(self.decoder.pass_grad, rate=rate)

    def encode(self, sentence):
        self.encoder.forward_pass(sentence)

    def decode(self, sentence=None):
        sample = random.choice(self.data)
        tree = sentence if sentence else sample.sentence2
        self.decoder.forward_pass(tree, self.encoder.get_root_embedding())

        return ' '.join([node.pword for node in self.decoder.tree])


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

        self.wxh = defaultdict(SquareZeros(self.dim))
        self.whh = self.random_weights(self.dim, self.dim)

        for dep in self.deps:
            self.wxh[dep] = self.random_weights(self.dim, self.dim)

    def compute_embeddings(self, node):
        # compute hidden state from input and previous hidden state
        hh_inp = np.dot(self.whh, self.prev)
        xh_inp = np.dot(self.wxh[node.dep_], self.xinp)

        # set hidden state and predictions as node attributes
        node.embedding = np.tanh(hh_inp + xh_inp)
        node.input = self.xinp

        node.py_w = self.softmax(np.dot(self.why_w, node.embedding))
        node.py_h = self.softmax(np.dot(self.why_h, node.embedding))
        node.py_d = self.softmax(np.dot(self.why_d, node.embedding))

        # set prediction targets as node attributes
        node.ty_w = self.get_word_target(node)
        node.ty_h = self.get_head_target(node)
        node.ty_d = self.get_dep_target(node)

        # set predicted words, heads, deps, as node attributes
        self.prev = node.embedding
        node.pw = self.idx_to_wrd[np.argmax(node.py_w, axis=0)[0]]
        node.ph = self.idx_to_wrd[np.argmax(node.py_h, axis=0)[0]]
        node.pd = self.idx_to_dep[np.argmax(node.py_d, axis=0)[0]]

        node.computed = True

    def predict_embedding(self):
        if len(self.sequence) == 0:
            xh_inp = np.dot(self.wxh['ROOT'], self.xinp)
        else:
            prev_node = self.sequence[-1]
            dep = prev_node.pd
            word = prev_node.pw
            head = prev_node.ph

            try:
                self.xinp = self.vectors[dep] + self.vectors[word]
                self.xinp += self.vectors[head]
            except KeyError:
                self.xinp = np.zeros_like(self.random_vector(self.dim))

            xh_inp = np.dot(self.wxh[dep], self.xinp)

        hh_inp = np.dot(self.whh, self.prev)

        node = Node('placeholder')
        node.embedding = np.tanh(hh_inp + xh_inp)
        node.input = self.xinp

        node.py_w = self.softmax(np.dot(self.why_w, node.embedding))
        node.py_h = self.softmax(np.dot(self.why_h, node.embedding))
        node.py_d = self.softmax(np.dot(self.why_d, node.embedding))

        self.prev = node.embedding
        node.pw = self.idx_to_wrd[np.argmax(node.py_w, axis=0)[0]]
        node.ph = self.idx_to_wrd[np.argmax(node.py_h, axis=0)[0]]
        node.pd = self.idx_to_dep[np.argmax(node.py_d, axis=0)[0]]

        node.computed = True
        self.sequence.append(node)

    def forward_pass(self, embedding, target):
        self.tree = [TokenWrapper(n) for n in self.parser(target)]
        self.root_embedding = embedding
        self.sequence = []

        self.prev = np.zeros_like(embedding)
        self.xinp = embedding

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
                    try:
                        ctx = [parent.head.lower_, parent.dep_, parent.lower_]
                        self.xinp = sum([np.copy(self.vectors[c]) for c in ctx
                                         if c in self.vectors])
                    except KeyError:
                        self.xinp = np.zeros_like(self.random_vector(self.dim))
                    self.compute_embeddings(node)
                    self.sequence.append(node)

        if len(self.sequence) == len(self.tree):
            return
        else:
            self.extend_sequence()

    def backward_pass(self, rate=0.01):
        dwxh = defaultdict(SquareZeros(self.dim))
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

    def predict(self, embedding, target_len):
        self.root_embedding = embedding
        self.sequence = []

        self.prev = np.zeros_like(embedding)
        self.xinp = embedding

        while True:
            self.predict_embedding()
            if len(self.sequence) == target_len:
                break

    def get_word_target(self, node):
        array = np.zeros_like(node.py_w)
        try:
            idx = self.wrd_to_idx[node.lower_]
            array[idx] = 1
        except KeyError:
            pass
        return array

    def get_head_target(self, node):
        array = np.zeros_like(node.py_h)
        try:
            idx = self.wrd_to_idx[node.head.lower_]
            array[idx] = 1
        except KeyError:
            pass
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
