import pickle
import spacy
import platform
import time

import numpy as np

from handlers import SNLI


class TokenWrapper(object):
    def __init__(self, token):
        self.token = token
        self.computed = False
        self._embedding = None

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, v):
        if not isinstance(v, np.ndarray):
            raise TypeError('Tree embeddings must be of type numpy.ndarray')
        else:
            self._embedding = v

    def __str__(self):
        return self.token.lower_

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.token, attr)


class DependencyNetwork(object):

    def __init__(self, embedding_dim, vocab, eps=0.1):
        self.dim = embedding_dim
        self.vocab = sorted(vocab)
        self.indices = {wrd: idx for idx, wrd in enumerate(self.vocab)}
        self.parser = spacy.load('en')
        self.vectors = np.random.random((len(vocab), self.dim))*eps*2-eps
        self.load_dependencies('depset.pickle')
        self.initialize_weights()

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def gaussian_id(dim):
        identity = np.eye(dim)
        gaussian = np.random.normal(loc=0, scale=0.05, size=(dim, dim))
        return identity + gaussian

    def load_dependencies(self, path):
        with open(path, 'rb') as pfile:
            self.depset = pickle.load(pfile)

    def initialize_weights(self):
        self.weights = {'token': self.gaussian_id(self.dim)}
        for dep in self.depset:
            self.weights[dep] = self.gaussian_id(self.dim)

    def forward_pass(self, sentence):
        self.tree = [TokenWrapper(t) for t in self.parser(sentence)]
        self.compute_leaves()
        self.compute_nodes()
        print(sentence, self.get_sentence_embedding())

    def backward_pass(self, error_signal):
        # root = self.get_tree_root()
        # children = self.get_children(root)
        pass

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
            idx = self.indices[token.lower_]
            emb = np.dot(self.weights['token'], self.vectors[idx, :])
        except KeyError:
            emb = np.zeros(self.dim)

        for child in children:
            emb += np.dot(self.weights[child.token.dep_], child.embedding)

        token.embedding = self.sigmoid(emb)
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

    def get_tree_root(self):
        for token in self.tree:
            if token.head.idx == token.idx:
                return token

    def get_sentence_embedding(self):
        for token in self.tree:
            if token.head.idx == token.idx:
                return token.embedding

if platform.system() == 'Linux':
    snlipath = '/home/pblouw/corpora/snli_1.0/'
    wikipath = '/home/pblouw/corpora/wikipedia'
    cachepath = '/home/pblouw/cache/'
else:
    snlipath = '/Users/peterblouw/corpora/snli_1.0/'
    wikipath = '/Users/peterblouw/corpora/wikipedia'
    cachepath = '/Users/peterblouw/cache/'


snli = SNLI(snlipath)
snli.extractor = snli.get_sentences
snli.load_vocab('snli_words')

model = DependencyNetwork(4, snli.vocab)

start_time = time.time()

for _ in range(5000):
    sentence = next(snli.train_data)[0]

    model.forward_pass(sentence)

print('Total runtime: ', time.time() - start_time)
# print(sentence)
# for token in model.tree:
#     # print(token._embedding)
#     if token.embedding is not None:
#         print(token, token.embedding)
