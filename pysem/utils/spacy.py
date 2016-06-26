import numpy as np


class TokenWrapper(object):
    def __init__(self, token):
        self.token = token
        self.computed = False
        self._embedding = None
        self._gradient = None

    @property
    def embedding(self):
        return self._embedding

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, grad):
        if not isinstance(grad, np.ndarray):
            raise TypeError('Tree gradients must be of type numpy.ndarray')
        else:
            self._gradient = grad

    @embedding.setter
    def embedding(self, vec):
        if not isinstance(vec, np.ndarray):
            raise TypeError('Tree embeddings must be of type numpy.ndarray')
        else:
            self._embedding = vec

    def __str__(self):
        return self.token.lower_

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.token, attr)
