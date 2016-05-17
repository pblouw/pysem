from mputils import apply_async, build_context_vectors

import numpy as np


class EmbeddingModel(object):
    """Base class for embedding models"""
    def __init__(self):
        pass

    @staticmethod
    def normalize(v):
        if np.linalg.norm(v) > 0:
            return v / np.linalg.norm(v)


class RandomIndexing(EmbeddingModel):

    def __init__(self, corpus):
        self._corpus = corpus

    def train(self, dimensions, vocab, preprocessing=None):
        self.dim = dimensions
        self.vocab = vocab

        base_vectors = {word: self.unit_vector() for word in vocab}
        context_vectors = {word: np.zeros(self.dim) for word in vocab}

        for _ in range(10):
            batch = next(self._corpus.batches)
            params = [(a, self.dim, base_vectors) for a in batch]
            for result in apply_async(build_context_vectors, params):
                for key in result.keys():
                    context_vectors[key] += result[key]

        return context_vectors

    def unit_vector(self):
        return self.normalize(np.random.normal(loc=0, size=1))


class SkipGram(EmbeddingModel):
    pass


class CBOW(EmbeddingModel):
    pass


class Word2Vec(SkipGram, CBOW):
    pass
