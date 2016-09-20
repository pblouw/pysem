import numpy as np

from pysem.utils.ml import LogisticRegression
from pysem.utils.snli import CompositeModel, bow_accuracy
from sklearn.feature_extraction.text import CountVectorizer


def test_bow_accuracy(snli):
    dim = 50
    snli.load_xy_pairs()
    data = [pair for pair in snli.train_data if pair.label != '-']

    vectorizer = CountVectorizer(binary=True)
    vectorizer.fit(snli.vocab)

    scale = 1 / np.sqrt(dim)
    size = (dim, len(vectorizer.get_feature_names()))
    bow_embedding_matrix = np.random.normal(loc=0, scale=scale, size=size)
    bow_classifier = LogisticRegression(n_features=2*dim, n_labels=3)

    acc = bow_accuracy(data, bow_classifier, bow_embedding_matrix, vectorizer)

    assert 0 < acc < 0.65


def test_rnn_accuracy(snli, rnn):
    classifier = LogisticRegression(n_features=2*rnn.dim, n_labels=3)
    model = CompositeModel(snli, rnn, classifier)
    model.acc.append(model.rnn_accuracy(model.dev_data))

    assert 0 < model.acc[0] < 0.65

    model.train(iters=2, bsize=1, rate=0.01, log_interval=2)
    assert len(model.acc) == 3


def test_dnn_accuracy(snli, dnn):
    classifier = LogisticRegression(n_features=2*dnn.dim, n_labels=3)
    model = CompositeModel(snli, dnn, classifier)
    model.acc.append(model.dnn_accuracy(model.dev_data))

    assert 0 < model.acc[0] < 0.65

    model.train(iters=2, bsize=1, rate=0.01, log_interval=2)
    assert len(model.acc) == 3


def test_average(snli):
    array = list(range(100))
    assert len(CompositeModel.average(array, [], 10)) == 10
