import os
import numpy as np

from pysem.handlers import Wikipedia
from pysem.embeddings import RandomIndexing

corpus_path = os.getcwd() + '/pysem/tests/corpora/'


def test_context_encoding():
    wp = Wikipedia(corpus_path, article_limit=10)
    wp.build_vocab(cutoff=0.05, batchsize=1)

    model = RandomIndexing(wp)
    model.train(dim=256, wordlist=wp.vocab, batchsize=1)

    assert isinstance(model.context_vectors, type(np.array(0)))
