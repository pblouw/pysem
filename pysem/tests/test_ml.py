import numpy as np

from pysem.utils.ml import LogisticRegression, MultiLayerPerceptron


def numerical_gradient(model, params, idx, xs, ys, delta=1e-5):
    val = params[idx]

    params[idx] = val + delta
    pcost = model.get_cost(xs, ys)

    params[idx] = val - delta
    ncost = model.get_cost(xs, ys)

    params[idx] = val
    numerical_gradient = (pcost - ncost) / (2 * delta)

    return numerical_gradient


def data_samples(di, do):
    xs = np.random.random((di, 2))
    ys = np.zeros((do, 2))
    ys[np.random.randint(0, do, 1), 0] = 1
    ys[np.random.randint(0, do, 1), 1] = 1

    return xs, ys


def test_logistic_regression():
    n_features = 50
    n_labels = 3
    n_gradient_checks = 50
    xs, ys = data_samples(n_features, n_labels)

    model = LogisticRegression(n_features=n_features, n_labels=n_labels)

    # Check weight gradients
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, model.weights.size, size=1)
        numerical = numerical_gradient(model, model.weights.flat, idx, xs, ys)
        model.train(xs, ys, rate=0.001)
        analytic = model.w_grad.flat[idx]

        assert np.allclose(numerical, analytic)

    # Check bias gradients
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, model.bias.size, size=1)
        numerical = numerical_gradient(model, model.bias.flat, idx, xs, ys)
        model.train(xs, ys, rate=0.001)
        analytic = model.b_grad.flat[idx]

        assert np.allclose(numerical, analytic)

    # Check that learning is successful
    for _ in range(50):
        model.train(xs, ys, rate=0.01)

    assert all(np.equal(model.predict(xs), np.argmax(ys, axis=0)))


def test_multi_layer_perceptron():
    di = 50
    dh = 25
    do = 3
    n_gradient_checks = 50
    xs, ys = data_samples(di, do)

    model = MultiLayerPerceptron(di=di, dh=dh, do=do)

    # Check input-to-hidden weight gradients
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, model.w1.size, size=1)
        numerical = numerical_gradient(model, model.w1.flat, idx, xs, ys)
        model.train(xs, ys, rate=0.001)
        analytic = model.w1_grad.flat[idx]

        assert np.allclose(numerical, analytic)

    # Check hidden-to-output weight gradients
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, model.w2.size, size=1)
        numerical = numerical_gradient(model, model.w2.flat, idx, xs, ys)
        model.train(xs, ys, rate=0.001)
        analytic = model.w2_grad.flat[idx]

        assert np.allclose(numerical, analytic)

    # Check hidden layer bias gradients
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, model.bh.size, size=1)
        numerical = numerical_gradient(model, model.bh.flat, idx, xs, ys)
        model.train(xs, ys, rate=0.001)
        analytic = model.bh_grad.flat[idx]

        assert np.allclose(numerical, analytic)

    # Check output layer bias gradients
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, model.bo.size, size=1)
        numerical = numerical_gradient(model, model.bo.flat, idx, xs, ys)
        model.train(xs, ys, rate=0.001)
        analytic = model.bo_grad.flat[idx]

        assert np.allclose(numerical, analytic)

    # Check that learning is successful
    for _ in range(100):
        model.train(xs, ys, rate=0.01)

    assert all(np.equal(model.predict(xs), np.argmax(ys, axis=0)))
