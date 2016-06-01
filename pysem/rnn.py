import pickle
import spacy
import platform
import time
import string

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
        self.embeddings = np.random.random((len(vocab), self.dim))*eps*2-eps
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

    def get_children(self, x):
        children = []
        for token in self.tree:
            if token.idx in [child.idx for child in x.children]:
                children.append(token)

        return children

    def compute_leaves(self):
        for token in self.tree:
            if len(list(token.children)) == 0:
                try:
                    idx = self.indices[token.lower_]
                except:
                    vector = np.zeros(self.dim)
                    token.embedding = vector
                    token.computed = True
                    continue

                vec = np.dot(self.weights['token'], self.embeddings[idx, :])
                token.embedding = self.sigmoid(vec)
                token.computed = True

    def compute_nodes(self, indices=[]):
        for token in self.tree:
            if len(list(token.children)) != 0 and not token.computed:
                children = self.get_children(token)
                computed = [c.computed for c in children]

                if all(computed):
                    try:
                        index = self.indices[token.lower_]
                    except:
                        vector = np.zeros(self.dim)
                        token.embedding = vector
                        token.computed = True
                        continue

                    vector = np.dot(self.weights['token'],
                                    self.embeddings[index, :])

                    for child in children:
                        vector += np.dot(self.weights[child.token.dep_],
                                         child.embedding)

                    token.embedding = self.sigmoid(vector)
                    token.computed = True

        computed = [t.computed for t in self.tree]
        if all(computed):
            return
        else:
            self.compute_nodes()


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

punc_translator = str.maketrans({key: None for key in string.punctuation})

model = DependencyNetwork(50, snli.vocab)

start_time = time.time()

for _ in range(50000):
    sentence = next(snli.train_data)[0]
    sentence = sentence.translate(punc_translator)
    print(sentence)
    model.forward_pass(sentence)

print('Total runtime: ', time.time() - start_time)
# print(sentence)
# for token in model.tree:
#     # print(token._embedding)
#     if token.embedding is not None:
#         print(token, token.embedding)
