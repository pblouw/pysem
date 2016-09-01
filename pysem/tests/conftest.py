import os
import pytest

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork, DependencyNetwork
from pysem.networks import HolographicNetwork
from pysem.generatives import EmbeddingGenerator, TreeGenerator
from pysem.lstm import LSTM, TreeLSTM

snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'


@pytest.fixture(scope='module')
def snli():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences
    snli.data = [x for x in snli.dev_data]
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


@pytest.fixture(scope='module')
def lstm(snli):
    dim = 50
    lstm = LSTM(dim=dim, vocab=snli.vocab)
    return lstm


@pytest.fixture(scope='module')
def treeLSTM(snli):
    dim = 50
    lstm = TreeLSTM(dim=dim, vocab=snli.vocab)
    return lstm


@pytest.fixture(scope='module')
def embedding_generator(snli):
    dim = 25
    data = [d for d in snli.train_data]

    # build subvocabs for each dependency
    subvocabs = {dep: set() for dep in DependencyNetwork.deps}

    for sample in data:
        s1_parse = DependencyNetwork.parser(sample.sentence1)
        s2_parse = DependencyNetwork.parser(sample.sentence2)

        for token in s1_parse:
            if token.lower_ not in subvocabs[token.dep_]:
                subvocabs[token.dep_].add(token.lower_)

        for token in s2_parse:
            if token.lower_ not in subvocabs[token.dep_]:
                subvocabs[token.dep_].add(token.lower_)

    return EmbeddingGenerator(dim=dim, subvocabs=subvocabs)


@pytest.fixture(scope='module')
def tree_generator(snli):
    dim = 50
    return TreeGenerator(dim=dim, vocab=snli.vocab)
