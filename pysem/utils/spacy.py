import numpy as np


class TokenWrapper(object):
    """A wrapper for spacy tokens so that they can be used as nodes in a
    computational graph defined by a dependency tree. Each spacy token links
    to its parent and children, and these links define the edges in the
    computational graph, which is then used to train a neural network with the
    same architecture. Each node in the graph corresponds to a layer in the
    resulting neural network.

    Parameters:
    ----------
    token : spacy token
        An instance of the type of object returned by the iterator made when
        spacy is called on a collection of text (e.g. a sentence or document)

    Attributes:
    ----------
    computed : boolean
        Whether or not the node corresponding to the wrapped token in the comp
        graph has been computed. Used for doing forward prop and backprop.

    embedding : numpy.ndarray
        The activation vector assigned to the corresponding node in the comp
        graph during forward propogation.
    gradient : numpy.ndarray
        The gradient of a cost function with respect to input into the
        corresponding node in the comp graph.
    """
    def __init__(self, token):
        self.token = token
        self._computed = False
        self._embedding = None
        self._gradient = None

    @property
    def computed(self):
        return self._computed

    @computed.setter
    def computed(self, val):
        if not isinstance(val, bool):
            raise TypeError('Use booleans to set whether a node is computed')
        else:
            self._computed = val

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, vec):
        if not isinstance(vec, np.ndarray):
            raise TypeError('Node embeddings must be of type numpy.ndarray')
        else:
            self._embedding = vec

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, grad):
        if not isinstance(grad, np.ndarray):
            raise TypeError('Node gradients must be of type numpy.ndarray')
        else:
            self._gradient = grad

    def __str__(self):
        return self.token.lower_

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self.token, attr)
