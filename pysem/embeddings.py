

class EmbeddingModel(object):
    """Base class for embedding models"""
    def __init__(self):
        pass


class RandomIndexing(EmbeddingModel):
    pass


class SkipGram(EmbeddingModel):
    pass


class CBOW(EmbeddingModel):
    pass


class Word2Vec(SkipGram, CBOW):
    pass
