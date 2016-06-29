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
