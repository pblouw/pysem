import random
import numpy as np

from collections import defaultdict
from pysem.embeddings import ContextEmbedding, OrderEmbedding, SyntaxEmbedding


def test_context_embedding(capfd, wikipedia):
    model = ContextEmbedding(wikipedia, wikipedia.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)

    sen_list = model.preprocess(next(model.corpus.articles))
    encoding = model.encode(sen_list)

    assert isinstance(sen_list, list)
    assert isinstance(''.join(random.choice(sen_list)), str)
    assert isinstance(encoding, defaultdict)
    assert isinstance(encoding[wikipedia.vocab[0]], np.ndarray)

    word_idx = model.word_to_idx[random.choice(model.vocab)]
    embedding = model.vectors[word_idx, :]

    n = 10
    matches = model.top_matches(embedding, n)
    assert len(matches) == n
    assert matches[0][1] > 0.999

    model.get_nearest(random.choice(model.vocab))
    printout, err = capfd.readouterr()
    assert isinstance(printout, str)


def test_order_embedding(capfd, wikipedia):
    model = OrderEmbedding(wikipedia, wikipedia.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)

    sen_list = model.preprocess(next(model.corpus.articles))
    encoding = model.encode(sen_list)

    assert isinstance(encoding, defaultdict)
    assert isinstance(encoding[wikipedia.vocab[0]], np.ndarray)

    model.get_completions(random.choice(model.vocab), position=1)
    model.get_completions(random.choice(model.vocab), position=-1)

    printout, err = capfd.readouterr()
    assert isinstance(printout, str)


def test_syntax_embedding(capfd, wikipedia):
    model = SyntaxEmbedding(wikipedia, wikipedia.vocab)
    model.train(dim=16, batchsize=1)

    assert isinstance(model.vectors, np.ndarray)

    encoding = model.encode(next(model.corpus.articles))

    assert isinstance(encoding, defaultdict)
    assert isinstance(encoding[wikipedia.vocab[0]], np.ndarray)

    model.get_verb_neighbors(random.choice(model.vocab), dep='nsubj')
    model.get_verb_neighbors(random.choice(model.vocab), dep='dobj')

    printout, err = capfd.readouterr()
    assert isinstance(printout, str)
