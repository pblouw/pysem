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
    eps = 0.5

    depnet = DependencyNetwork(dim=dim, vocab=snli.vocab)
    sample = next(snli.train_data)
    sen = random.choice(sample)

    error_grad = np.random.random((dim, 1)) * 2 * eps - eps

    depnet.forward_pass(sen)
    deps = [node.dep_ for node in depnet.tree if node.dep_ != 'ROOT']

    # Save a copy of the weights before SGD update
    weights = []
    for dep in deps:
        weights.append(np.copy(depnet.weights[dep]))

    # Do backprop
    depnet.backward_pass(error_grad, rate=0.1)

    # Check that a gradient is computed for every node in the tree
    for node in depnet.tree:
        assert isinstance(node.gradient, np.ndarray)

    # Check that gradient norms are nonzero
    for dep in deps:
        assert np.linalg.norm(depnet.wgrads[dep].flatten()) != 0

    # Save a copy of the weights after SGD update
    new_weights = []
    for dep in deps:
        new_weights.append(np.copy(depnet.weights[dep]))

    # Check that every weight has changed after the SGD update
    for pair in zip(weights, new_weights):
        assert np.count_nonzero(pair[1] - pair[0]) == pair[0].size
