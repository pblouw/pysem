import os
import string

import numpy as np

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork
from pysem.utils.ml import LogisticRegression

strip_pun = str.maketrans({key: None for key in string.punctuation})

snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'


def get_cost(model, logreg, xs, ys):
    model.forward_pass(xs)
    embedding = model.get_root_embedding()

    return logreg.get_cost(embedding, ys)


def num_grad(model, params, idx, xs, ys, logreg, delta=1e-5):
    val = np.copy(params[idx])

    params[idx] = val + delta
    pcost = get_cost(model, logreg, xs, ys)

    params[idx] = val - delta
    ncost = get_cost(model, logreg, xs, ys)

    params[idx] = val
    numerical_gradient = (pcost - ncost) / (2 * delta)

    return numerical_gradient


def test_forward_pass():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50

    rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    sens = [next(snli.train_data) for _ in range(5)]
    sens = [item.sentence1 for item in sens]

    rnn.forward_pass(sens)
    sen_vec = rnn.get_root_embedding()

    assert isinstance(sen_vec, np.ndarray)


def test_backward_pass():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50
    eps = 0.5

    rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    sens = [next(snli.train_data) for _ in range(5)]
    sens = [item.sentence1 for item in sens]

    error_grad = np.random.random((dim, 5)) * 2 * eps - eps

    rnn.forward_pass(sens)

    # Save a copy of the weights before SGD update
    weights = np.copy(rnn.whh)

    # Do backprop
    rnn.backward_pass(error_grad, rate=0.1)

    new_weights = np.copy(rnn.whh)

    # Check that every weight has changed after the SGD update
    assert np.count_nonzero(weights - new_weights) == weights.size


def test_weight_gradients():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    xs = [next(snli.train_data) for _ in range(5)]
    xs = [item.sentence1 for item in xs]
    ys = np.zeros((n_labels, 5))
    ys[np.random.randint(0, n_labels, 5), list(range(5))] = 1

    rnn.forward_pass(xs)

    # Use random weight in each matrix for n numerical gradient checks
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, rnn.whh.size, size=1)
        params = rnn.whh.flat

        numerical = num_grad(rnn, params, idx, xs, ys, logreg)

        rnn.forward_pass(xs)

        logreg.train(rnn.get_root_embedding(), ys, rate=0.001)

        rnn.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = rnn.dwhh.flat[idx]

        assert np.allclose(analytic, numerical)


def test_embedding_gradients():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    xs = [next(snli.train_data) for _ in range(1)]
    xs = [item.sentence1 for item in xs]
    ys = np.zeros((n_labels, 1))
    ys[np.random.randint(0, n_labels, 1), list(range(1))] = 1

    rnn.forward_pass(xs)
    words = xs[0].split()
    words = [word.translate(strip_pun) for word in words]
    words = [word.lower() for word in words]

    # Use random element in each word embedding for n numerical gradient checks
    for _ in range(n_gradient_checks):
        for word in words:
            print(word)
            idx = np.random.randint(0, rnn.vectors[word].size, size=1)
            params = rnn.vectors[word].flat

            numerical = num_grad(rnn, params, idx, xs, ys, logreg)

            rnn.forward_pass(xs)

            logreg.train(rnn.get_root_embedding(), ys, rate=0.001)

            rnn.backward_pass(logreg.yi_grad, rate=0.001)
            analytic = rnn.dwrd[word].flat[idx]

            assert np.allclose(analytic, numerical)


def test_bias_gradients():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    rnn = RecurrentNetwork(dim=dim, vocab=snli.vocab)
    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    xs = [next(snli.train_data) for _ in range(5)]
    xs = [item.sentence1 for item in xs]
    ys = np.zeros((n_labels, 5))
    ys[np.random.randint(0, n_labels, 5), list(range(5))] = 1

    rnn.forward_pass(xs)

    # Use random weight in each matrix for n numerical gradient checks
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, rnn.bh.size, size=1)
        params = rnn.bh.flat

        numerical = num_grad(rnn, params, idx, xs, ys, logreg)

        rnn.forward_pass(xs)

        logreg.train(rnn.get_root_embedding(), ys, rate=0.001)

        rnn.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = rnn.dbh.flat[idx]

        assert np.allclose(analytic, numerical)

        idx = np.random.randint(0, rnn.by.size, size=1)
        params = rnn.by.flat

        numerical = num_grad(rnn, params, idx, xs, ys, logreg)

        rnn.forward_pass(xs)

        logreg.train(rnn.get_root_embedding(), ys, rate=0.001)

        rnn.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = rnn.dby.flat[idx]

        assert np.allclose(analytic, numerical)
