import os
import pytest

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork, DependencyNetwork
from pysem.networks import HolographicNetwork

snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'


@pytest.fixture(scope='module')
def snli():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences
    return snli


@pytest.fixture(scope='module')
def rnn(snli):
    dim = 50
    rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    return rnn


@pytest.fixture(scope='module')
def dnn(snli):
    dim = 50
    dnn = DependencyNetwork(dim=dim, vocab=snli.vocab)
    return dnn


@pytest.fixture(scope='module')
def hnn(snli):
    dim = 50
    hnn = HolographicNetwork(dim=dim, vocab=snli.vocab)
    return hnn