import pickle
import numpy as np


class DepRNN(object):

    def __init__(self, embedding_dim):
        self.dim = embedding_dim
        self.load_dependencies('dependencies.pickle')
        self.initialize_weights()

    @staticmethod
    def gaussian_id(dim):
        identity = np.eye(dim)
        gaussian = np.random.normal(loc=0, scale=0.05, size=(dim, dim))
        return identity + gaussian

    def load_dependencies(self, path):
        with open(path, 'rb') as pfile:
            self.depset = pickle.load(pfile)

    def initialize_weights(self):
        self.weights = dict()
        for dep in self.depset:
            self.weights[dep] = self.gaussian_id(self.dim)

    def forward_pass(self, sentence):
        pass

    def backward_pass(self, error_sig):
        pass

    def build_tree(self, sentence):
        pass
