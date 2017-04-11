from pysem.utils.ml import LogisticRegression
from pysem.utils.snli import CompositeModel, BagOfWords


def test_bow_accuracy(snli):
    dim = 50
    encoder = BagOfWords(dim=dim, vocab=snli.vocab)
    classifier = LogisticRegression(n_features=2*dim, n_labels=3)

    model = CompositeModel(snli, encoder, classifier)
    model.train(iters=2, bsize=1, rate=0.01, log_interval=2)

    assert len(model.acc) == 2
    assert 0 < model.acc[0] < 0.65


def test_rnn_accuracy(snli, rnn):
    classifier = LogisticRegression(n_features=2*rnn.dim, n_labels=3)

    model = CompositeModel(snli, rnn, classifier)
    model.train(iters=2, bsize=1, rate=0.01, log_interval=2)

    assert len(model.acc) == 2
    assert 0 < model.acc[0] < 0.65


def test_dnn_accuracy(snli, dnn):
    classifier = LogisticRegression(n_features=2*dnn.dim, n_labels=3)

    model = CompositeModel(snli, dnn, classifier)
    model.train(iters=2, bsize=1, rate=0.01, log_interval=2)

    assert len(model.acc) == 2
    assert 0 < model.acc[0] < 0.65


def test_predict(snli, dnn, rnn):
    labels = {'entailment', 'contradiction', 'neutral'}

    classifier = LogisticRegression(n_features=2*dnn.dim, n_labels=3)
    model = CompositeModel(snli, dnn, classifier)
    model.train(iters=2, bsize=1, rate=0.01, log_interval=2)

    label = model.predict(s1='The boy ran.', s2='The boy moved.')
    assert label in labels

    model = CompositeModel(snli, rnn, classifier)
    model.train(iters=2, bsize=1, rate=0.01, log_interval=2)

    label = model.predict(s1='The boy ran.', s2='The boy moved.')
    assert label in labels


def test_average(snli):
    array = list(range(100))
    assert len(CompositeModel.average(array, [], 10)) == 10
