import os
import pytest
import numpy as np

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork, DependencyNetwork
from pysem.networks import HolographicNetwork
from pysem.generatives import EmbeddingGenerator, TreeGenerator
from pysem.utils.ml import LogisticRegression
from pysem.lstm import LSTM, TreeLSTM

snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'
dim = 50
n_labels = 3


def cost(model, classifier, xs, ys):
    model.forward_pass(xs)
    embedding = model.get_root_embedding()

    return classifier.get_cost(embedding, ys)


def ngrad(model, params, idx, xs, ys, classifier, get_cost=cost, delta=1e-6):
    val = np.copy(params[idx])

    params[idx] = val + delta
    pcost = get_cost(model, classifier, xs, ys)
    params[idx] = val - delta
    ncost = get_cost(model, classifier, xs, ys)
    params[idx] = val

    numerical_gradient = (pcost - ncost) / (2 * delta)
    return numerical_gradient


@pytest.fixture(scope='session')
def get_cost():
    return cost


@pytest.fixture(scope='session')
def num_grad():
    return ngrad


@pytest.fixture(scope='module')
def snli():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences
    snli.data = [x for x in snli.dev_data]
    return snli


@pytest.fixture(scope='module')
def classifier():
    classifier = LogisticRegression(n_features=dim, n_labels=n_labels)
    return classifier


@pytest.fixture(scope='module')
def rnn(snli):
    rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    return rnn


@pytest.fixture(scope='module')
def dnn(snli):
    dnn = DependencyNetwork(dim=dim, vocab=snli.vocab)
    return dnn


@pytest.fixture(scope='module')
def hnn(snli):
    hnn = HolographicNetwork(dim=dim, vocab=snli.vocab)
    return hnn


@pytest.fixture(scope='module')
def lstm(snli):
    lstm = LSTM(input_dim=dim, cell_dim=dim*2, vocab=snli.vocab)
    return lstm


@pytest.fixture(scope='module')
def treeLSTM(snli):
    lstm = TreeLSTM(input_dim=dim, cell_dim=dim*2, vocab=snli.vocab)
    return lstm


@pytest.fixture(scope='module')
def embgen(snli):
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
def tree_gen(snli):
    return TreeGenerator(dim=dim, vocab=snli.vocab)
