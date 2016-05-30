import os
import numpy as np

from pysem.handlers import Wikipedia
from pysem.embeddings import RandomIndexing

corpus_path = os.getcwd() + '/pysem/tests/corpora/'


def test_context_encoding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(cutoff=0.05, batchsize=1)

    model = RandomIndexing(wp)
    model.train(dim=16, wordlist=wp.vocab, flags=['context'], batchsize=1)

    assert isinstance(model.context_vectors, type(np.array(0)))


def test_order_encoding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(cutoff=0.05, batchsize=1)

    model = RandomIndexing(wp)
    model.train(dim=16, wordlist=wp.vocab, flags=['order'], batchsize=1)

    assert isinstance(model.order_vectors, type(np.array(0)))


def test_syntax_encoding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(cutoff=0.05, batchsize=1)

    model = RandomIndexing(wp)
    model.train(dim=16, wordlist=wp.vocab, flags=['syntax'], batchsize=1)

    assert isinstance(model.syntax_vectors, type(np.array(0)))


def test_default_encoding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(cutoff=0.05, batchsize=1)

    model = RandomIndexing(wp)
    model.train(dim=16, wordlist=wp.vocab, batchsize=1)

    assert isinstance(model.syntax_vectors, type(np.array(0)))
    assert isinstance(model.order_vectors, type(np.array(0)))
    assert isinstance(model.context_vectors, type(np.array(0)))
