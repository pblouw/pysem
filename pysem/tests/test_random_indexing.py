import os
import random
import numpy as np

from collections import defaultdict
from pysem.corpora import Wikipedia
from pysem.embeddings import ContextEmbedding, OrderEmbedding, SyntaxEmbedding

corpus_path = os.getcwd() + '/pysem/tests/corpora/wikipedia/'


def test_context_embedding():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(threshold=0.1, batchsize=1)

    model = ContextEmbedding(wp, wp.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)

    model.corpus.reset_streams()
    sen_list = model.preprocess(next(model.corpus.articles))
    encoding = model.encode(sen_list)

    assert isinstance(sen_list, list)
    assert isinstance(''.join(random.choice(sen_list)), str)
    assert isinstance(encoding, defaultdict)
    assert isinstance(encoding[wp.vocab[0]], np.ndarray)

    word_idx = model.word_to_idx[random.choice(model.vocab)]
    embedding = model.vectors[word_idx, :]

    n = 10
    matches = model.top_matches(embedding, n)
    assert len(matches) == n
    assert matches[0][1] > 0.999


def test_order_embedding(capfd):
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(threshold=0.1, batchsize=1)

    model = OrderEmbedding(wp, wp.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)

    model.corpus.reset_streams()
    sen_list = model.preprocess(next(model.corpus.articles))
    encoding = model.encode(sen_list)

    assert isinstance(encoding, defaultdict)
    assert isinstance(encoding[wp.vocab[0]], np.ndarray)

    model.get_completions(random.choice(model.vocab), position=1)
    model.get_completions(random.choice(model.vocab), position=-1)

    printout, err = capfd.readouterr()
    assert isinstance(printout, str)


def test_syntax_embedding(capfd):
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(threshold=0.1, batchsize=1)

    model = SyntaxEmbedding(wp, wp.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)

    model.corpus.reset_streams()
    encoding = model.encode(next(model.corpus.articles))

    assert isinstance(encoding, defaultdict)
    assert isinstance(encoding[wp.vocab[0]], np.ndarray)

    model.get_verb_neighbors(random.choice(model.vocab), dep='nsubj')
    model.get_verb_neighbors(random.choice(model.vocab), dep='dobj')

    printout, err = capfd.readouterr()
    assert isinstance(printout, str)

