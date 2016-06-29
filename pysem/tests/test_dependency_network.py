import os
import random
import pytest

import numpy as np

from pysem.corpora import SNLI
from pysem.networks import DependencyNetwork

snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'


def test_token_wrapper():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50

    depnet = DependencyNetwork(dim=dim, vocab=snli.vocab)
    sample = next(snli.train_data)
    sen = random.choice(sample)

    depnet.forward_pass(sen)

    node = random.choice(depnet.tree)

    with pytest.raises(TypeError):
        node.embedding = ''
        node.gradient = ''
        node.computed = ''

    assert node.computed is False
    assert node.gradient is None
    assert node.__str__() in depnet.vocab

    assert node.dep_ in depnet.deps


def test_forward_pass():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50

    depnet = DependencyNetwork(dim=dim, vocab=snli.vocab)
    sample = next(snli.train_data)
    sen = random.choice(sample)

    depnet.forward_pass(sen)
    sen_vec = depnet.get_root_embedding()

    assert isinstance(sen_vec, np.ndarray)

    for node in depnet.tree:
        assert isinstance(node.embedding, np.ndarray)


def test_backward_pass():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50
    eps = 0.1

    depnet = DependencyNetwork(dim=dim, vocab=snli.vocab)
    sample = next(snli.train_data)
    sen = random.choice(sample)

    error_grad = np.random.random((dim, 1)) * 2 * eps - eps

    depnet.forward_pass(sen)
    deps = [node.dep_ for node in depnet.tree]

    weights = []
    for dep in deps:
        weights.append(depnet.weights[dep])

    depnet.backward_pass(error_grad, rate=0.1)

    for node in depnet.tree:
        assert isinstance(node.gradient, np.ndarray)

    new_weights = []
    for dep in deps:
        new_weights.append(depnet.weights[dep])

    differences = [w[1]-w[0] for w in zip(weights, new_weights)]

    for difference in differences:
        assert np.count_nonzero(difference) == 0
