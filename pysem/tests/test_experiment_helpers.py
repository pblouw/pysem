import os

import numpy as np

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork, DependencyNetwork
from pysem.utils.ml import LogisticRegression
from pysem.utils.experiments import bow_accuracy, rnn_accuracy, dnn_accuracy
from sklearn.feature_extraction.text import CountVectorizer

snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'
dim = 50


def test_bow_accuracy():
    snli = SNLI(snli_path)
    snli.build_vocab()

    snli.extractor = snli.get_xy_pairs
    data = [pair for pair in snli.train_data if pair.label != '-']

    vectorizer = CountVectorizer(binary=True)
    vectorizer.fit(snli.vocab)

    scale = 1 / np.sqrt(dim)
    size = (dim, len(vectorizer.get_feature_names()))
    bow_embedding_matrix = np.random.normal(loc=0, scale=scale, size=size)
    bow_classifier = LogisticRegression(n_features=2*dim, n_labels=3)

    acc = bow_accuracy(data, bow_classifier, bow_embedding_matrix, vectorizer)

    assert 0 < acc < 0.65


def test_rnn_accuracy():
    snli = SNLI(snli_path)
    snli.build_vocab()

    snli.extractor = snli.get_xy_pairs
    data = [pair for pair in snli.train_data if pair.label != '-']

    s1_rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    s2_rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    rnn_classifier = LogisticRegression(n_features=2*dim, n_labels=3)

    acc = rnn_accuracy(data, rnn_classifier, s1_rnn, s2_rnn)

    assert 0 < acc < 0.65


def test_dnn_accuracy():
    snli = SNLI(snli_path)
    snli.build_vocab()

    snli.extractor = snli.get_xy_pairs
    data = [pair for pair in snli.train_data if pair.label != '-']

    s1_dnn = DependencyNetwork(dim=dim, vocab=snli.vocab)
    s2_dnn = DependencyNetwork(dim=dim, vocab=snli.vocab)
    dnn_classifier = LogisticRegression(n_features=2*dim, n_labels=3)

    acc = dnn_accuracy(data, dnn_classifier, s1_dnn, s2_dnn)

    assert 0 < acc < 0.65
