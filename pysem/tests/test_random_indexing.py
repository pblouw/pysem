import os
import numpy as np

from pysem.corpora import Wikipedia
from pysem.embeddings import ContextEmbedding, OrderEmbedding, SyntaxEmbedding

corpus_path = os.getcwd() + '/pysem/tests/corpora/wikipedia/'


def test_context_embedding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(threshold=0.1, batchsize=1)

    model = ContextEmbedding(wp, wp.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)


def test_order_embedding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(threshold=0.1, batchsize=1)

    model = OrderEmbedding(wp, wp.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)


def test_syntax_embedding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(threshold=0.1, batchsize=1)

    model = SyntaxEmbedding(wp, wp.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)
