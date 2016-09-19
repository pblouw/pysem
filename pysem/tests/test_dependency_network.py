import random
import pytest
import numpy as np

n_gradient_checks = 25
n_labels = 3
rate = 0.001


def random_data(snli):
    sample = random.choice(snli.train_data)
    xs = random.choice(sample)
    ys = np.zeros(n_labels)
    ys[np.random.randint(0, n_labels, 1)] = 1
    ys = ys.reshape(n_labels, 1)

    return xs, ys


def train_step(s_model, classifier, xs, ys):
    s_model.forward_pass(xs)
    classifier.train(s_model.get_root_embedding(), ys, rate=rate)
    s_model.backward_pass(classifier.yi_grad, rate=rate)


def test_token_wrapper(dnn, snli):
    sample = random.choice(snli.train_data)
    sen = random.choice(sample)
    dnn.forward_pass(sen)

    node = random.choice(dnn.tree)

    with pytest.raises(TypeError):
        node.embedding = ''
        node.gradient = ''
        node.computed = ''

    assert node.computed is False
    assert node.gradient is None
    assert node.dep_ in dnn.deps


def test_forward_pass(dnn, snli):
    sample = random.choice(snli.train_data)
    sen = random.choice(sample)
    dnn.forward_pass(sen)
    sen_vec = dnn.get_root_embedding()

    assert isinstance(sen_vec, np.ndarray)

    for node in dnn.tree:
        assert isinstance(node.embedding, np.ndarray)


def test_backward_pass(dnn, snli):
    sample = random.choice(snli.train_data)
    sen = random.choice(sample)

    error_grad = np.random.random((dnn.dim, 1)) * 2 * 0.5 - 0.5

    dnn.forward_pass(sen)
    deps = [node.dep_ for node in dnn.tree if node.dep_ != 'ROOT']

    # Save a copy of the weights before SGD update
    weights = []
    for dep in deps:
        weights.append(np.copy(dnn.weights[dep]))

    # Do backprop
    dnn.backward_pass(error_grad, rate=rate)

    # Check that a gradient is computed for every node in the tree
    for node in dnn.tree:
        assert isinstance(node.gradient, np.ndarray)

    # Check that gradient norms are nonzero
    for dep in deps:
        assert np.linalg.norm(dnn.wgrads[dep].flatten()) != 0

    # Save a copy of the weights after SGD update
    new_weights = []
    for dep in deps:
        new_weights.append(np.copy(dnn.weights[dep]))

    # Check that every weight has changed after the SGD update
    for pair in zip(weights, new_weights):
        assert np.count_nonzero(pair[1] - pair[0]) == pair[0].size


def test_weight_gradients(dnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    dnn.forward_pass(xs)
    deps = [node.dep_ for node in dnn.tree if node.dep_ != 'ROOT']

    # Use random weight in each matrix for n numerical gradient checks
    for _ in range(n_gradient_checks):
        for dep in deps:
            idx = np.random.randint(0, dnn.weights[dep].size, size=1)
            params = dnn.weights[dep].flat

            numerical_grad = num_grad(dnn, params, idx, xs, ys, classifier)
            train_step(dnn, classifier, xs, ys)
            analytic_grad = dnn.wgrads[dep].flat[idx]

            assert np.allclose(analytic_grad, numerical_grad)


def test_embedding_gradients(dnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    dnn.forward_pass(xs)
    words = [node.lower_ for node in dnn.tree]
    words = [w for w in words if w in snli.vocab]

    for _ in range(n_gradient_checks):
        for word in words:
            idx = np.random.randint(0, dnn.vectors[word].size, size=1)
            params = dnn.vectors[word].flat

            numerical_grad = num_grad(dnn, params, idx, xs, ys, classifier)
            train_step(dnn, classifier, xs, ys)
            analytic_grad = dnn.xgrads[word].flat[idx]

            assert np.allclose(analytic_grad, numerical_grad)


def test_bias_gradients(dnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    dnn.forward_pass(xs)
    deps = [node.dep_ for node in dnn.tree if node.dep_ != 'ROOT']

    for _ in range(n_gradient_checks):
        for dep in deps:
            idx = np.random.randint(0, dnn.biases[dep].size, size=1)
            params = dnn.biases[dep].flat

            numerical_grad = num_grad(dnn, params, idx, xs, ys, classifier)
            train_step(dnn, classifier, xs, ys)
            analytic_grad = dnn.bgrads[dep].flat[idx]

            assert np.allclose(analytic_grad, numerical_grad)


def test_wm_gradients(dnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, dnn.wm.size, size=1)
        params = dnn.wm.flat

        numerical_grad = num_grad(dnn, params, idx, xs, ys, classifier)
        train_step(dnn, classifier, xs, ys)
        analytic_grad = dnn.dwm.flat[idx]

        assert np.allclose(analytic_grad, numerical_grad)
