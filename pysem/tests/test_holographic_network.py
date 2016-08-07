import os
import random

import numpy as np

from pysem.corpora import SNLI
from pysem.networks import HolographicNetwork
from pysem.utils.ml import LogisticRegression

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


def test_forward_pass(hnn, snli):
    sample = next(snli.train_data)
    sen = random.choice(sample)

    hnn.forward_pass(sen)
    sen_vec = hnn.get_root_embedding()

    assert isinstance(sen_vec, np.ndarray)

    for node in hnn.tree:
        assert isinstance(node.embedding, np.ndarray)


def test_backward_pass(hnn, snli):
    dim = 50
    eps = 0.5

    sample = next(snli.train_data)
    sen = random.choice(sample)

    error_grad = np.random.random((dim, 1)) * 2 * eps - eps

    hnn.forward_pass(sen)
    words = [node.lower_ for node in hnn.tree if node.lower_ in snli.vocab]

    # Save a copy of the word vectors before SGD update
    vectors = []
    for word in words:
        vectors.append(np.copy(hnn.vectors[word]))

    # Do backprop
    hnn.backward_pass(error_grad, rate=0.1)

    # Check that a gradient is computed for every word vector
    for node in hnn.tree:
        assert isinstance(node.gradient, np.ndarray)

    # Check that gradient norms are nonzero
    for word in words:
        assert np.linalg.norm(hnn.vectors[word]) != 0

    # Save a copy of the word vectors after SGD update
    new_vectors = []
    for word in words:
        new_vectors.append(np.copy(hnn.vectors[word]))

    # Check that every word vector has changed after the SGD update
    for pair in zip(vectors, new_vectors):
        assert np.count_nonzero(pair[1] - pair[0]) == pair[0].size


def test_embedding_gradients(hnn, snli):
    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    sample = next(snli.train_data)
    xs = random.choice(sample)
    ys = np.zeros(n_labels)
    ys[np.random.randint(0, n_labels, 1)] = 1
    ys = ys.reshape(n_labels, 1)

    hnn.forward_pass(xs)
    words = [node.lower_ for node in hnn.tree if node.lower_ in snli.vocab]

    for _ in range(n_gradient_checks):
        for word in words:
            idx = np.random.randint(0, hnn.vectors[word].size, size=1)
            params = hnn.vectors[word].flat

            numerical = num_grad(hnn, params, idx, xs, ys, logreg)

            hnn.forward_pass(xs)

            logreg.train(hnn.get_root_embedding(), ys, rate=0.001)

            error_grad = logreg.yi_grad

            hnn.backward_pass(error_grad, rate=0.001)
            node = [node for node in hnn.tree if node.lower_ == word].pop()
            analytic = node.gradient.flat[idx]

            assert np.allclose(analytic, numerical)
