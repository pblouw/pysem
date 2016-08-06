import os
import random
import pytest

import numpy as np

from pysem.corpora import SNLI
from pysem.networks import HolographicNetwork, RecurrentNetwork 
from pysem.networks import DependencyNetwork
from pysem.utils.ml import LogisticRegression


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