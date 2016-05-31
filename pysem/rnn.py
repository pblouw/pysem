import pickle
import spacy
import platform
import numpy as np

from handlers import SNLI


class DependencyNode(object):
    def __init__(self, token):
        self.token = token
        self.embedding = None
        self.computed = False


class DependencyNetwork(object):

    def __init__(self, embedding_dim, vocab, eps=0.1):
        self.dim = embedding_dim
        self.vocab = sorted(vocab)
        self.indices = {wrd: idx for idx, wrd in enumerate(self.vocab)}
        self.parser = spacy.load('en')
        self.embeddings = np.random.random((len(vocab), self.dim))*eps*2-eps
        self.load_dependencies('dependencies.pickle')
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
        self.parse = self.parser(sentence)
        self.nodes = {t: DependencyNode(t) for t in self.parse}

        for token in self.parse:
            if list(token.subtree) == [token]:
                node = self.nodes[token]
                index = self.indices[token.lower_]

                leaf_embedding = self.sigmoid(np.dot(self.weights['token'],
                                              self.embeddings[index, :]))

                node.embedding = leaf_embedding
                node.computed = True

                print(token, ' Added Embedding!')

        for token in self.parse:
            if list(token.subtree) != [token]:
                children = list(token.children)
                children = [self.nodes[child] for child in children]

                computed = [c.computed for c in children]
                if all(computed):
                    # print(token, list(token.children), computed)

                    index = self.indices[token.lower_]
                    mapping = np.dot(self.weights['token'],
                                     self.embeddings[index, :])
                    for child in children:
                        mapping += np.dot(self.weights[node.token.dep_],
                                          node.embedding)

                    node = self.nodes[token]
                    node.embedding = self.sigmoid(mapping)
                    node.computed = True
                    print(node.token, [c.embedding for c in children],
                          [c.token.dep_ for c in children])

    def compute_available(self, parse):
        pass


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
# snli.build_vocab()
snli.load_vocab('snli_words')

sentence = next(snli.train_data)[0]

model = DependencyNetwork(4, snli.vocab)
model.forward_pass(sentence)

print(sentence)
for node in model.nodes.values():
    if node.embedding is not None:
        print(node.token)
