import random

import numpy as np

n_gradient_checks = 25
n_labels = 3
bsize = 10
rate = 0.001


def random_data(snli):
    samples = [random.choice(snli.data) for _ in range(bsize)]
    xs = [random.choice(sample) for sample in samples]
    ys = np.zeros((n_labels, bsize))
    ys[np.random.randint(0, n_labels, bsize), list(range(bsize))] = 1

    return xs, ys


def train_step(s_model, classifier, xs, ys):
    s_model.forward_pass(xs)
    classifier.train(s_model.get_root_embedding(), ys, rate=rate)
    s_model.backward_pass(classifier.yi_grad, rate=rate)


def test_forward_pass(rnn, snli):
    xs, ys = random_data(snli)
    rnn.forward_pass(xs)

    assert isinstance(rnn.get_root_embedding(), np.ndarray)


def test_backward_pass(rnn, snli):
    xs, ys = random_data(snli)

    error_grad = np.random.random((rnn.dim, bsize)) * 2 * 0.5 - 0.5

    rnn.forward_pass(xs)

    # Save a copy of the weights before SGD update
    weights = np.copy(rnn.whh)

    # Do backprop
    rnn.backward_pass(error_grad, rate=rate)

    # Save a copy of the weights after SGD update
    new_weights = np.copy(rnn.whh)

    # Check that every weight has changed across the update
    assert np.count_nonzero(weights - new_weights) == weights.size


def test_weight_gradients(rnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # Use random weight in each matrix for n numerical gradient checks
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, rnn.whh.size, size=1)
        params = rnn.whh.flat

        numerical_grad = num_grad(rnn, params, idx, xs, ys, classifier)
        train_step(rnn, classifier, xs, ys)
        analytic_grad = rnn.dwhh.flat[idx]

        assert np.allclose(analytic_grad, numerical_grad)

        idx = np.random.randint(0, rnn.why.size, size=1)
        params = rnn.why.flat

        numerical_grad = num_grad(rnn, params, idx, xs, ys, classifier)
        train_step(rnn, classifier, xs, ys)
        analytic_grad = rnn.dwhy.flat[idx]

        assert np.allclose(analytic_grad, numerical_grad)


def test_embedding_gradients(rnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    rnn.forward_pass(xs)
    words = rnn.batch[0]
    words = [w.lower() for w in words if w != 'PAD']

    # Use random element in each word embedding for n numerical gradient checks
    for _ in range(n_gradient_checks):
        for word in words:
            print(word)
            idx = np.random.randint(0, rnn.vectors[word].size, size=1)
            params = rnn.vectors[word].flat

            numerical_grad = num_grad(rnn, params, idx, xs, ys, classifier)
            train_step(rnn, classifier, xs, ys)
            analytic_grad = rnn.xgrads[word].flat[idx]

            assert np.allclose(analytic_grad, numerical_grad)


def test_bias_gradients(rnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # Use random element in each bias vector for n numerical gradient checks
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, rnn.bh.size, size=1)
        params = rnn.bh.flat

        numerical_grad = num_grad(rnn, params, idx, xs, ys, classifier)
        train_step(rnn, classifier, xs, ys)
        analytic_grad = rnn.dbh.flat[idx]

        if not rnn.clipflag:
            assert np.allclose(analytic_grad, numerical_grad)

        idx = np.random.randint(0, rnn.by.size, size=1)
        params = rnn.by.flat

        numerical_grad = num_grad(rnn, params, idx, xs, ys, classifier)
        train_step(rnn, classifier, xs, ys)
        analytic_grad = rnn.dby.flat[idx]

        assert np.allclose(analytic_grad, numerical_grad)
