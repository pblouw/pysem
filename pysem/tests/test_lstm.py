import random
import numpy as np

from pysem.utils.ml import LogisticRegression


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


def test_forward_pass(lstm, snli):
    sample = next(snli.train_data)
    sen = random.choice(sample)

    lstm.forward_pass(sen)
    sen_vec = lstm.get_root_embedding()

    assert isinstance(sen_vec, np.ndarray)

    for i in range(len(lstm.sen)):
        assert isinstance(lstm.i_gates[i], np.ndarray)
        assert isinstance(lstm.f_gates[i], np.ndarray)
        assert isinstance(lstm.o_gates[i], np.ndarray)
        assert isinstance(lstm.cell_inputs[i], np.ndarray)
        assert isinstance(lstm.cell_states[i], np.ndarray)
        assert isinstance(lstm.hs[i], np.ndarray)


def test_gate_weight_gradients(lstm, snli):
    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    sample = random.choice(snli.data)
    xs = random.choice(sample)
    ys = np.zeros(n_labels)
    ys[np.random.randint(0, n_labels, 1)] = 1
    ys = ys.reshape(n_labels, 1)

    # test input to forget gate weights
    for _ in range(n_gradient_checks):
        print(_)
        idx = np.random.randint(0, lstm.oW.size, size=1)
        params = lstm.oW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.doW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to output gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.fW.size, size=1)
        params = lstm.fW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.dfW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to input gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.iW.size, size=1)
        params = lstm.iW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.diW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test prev state to forget gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.fU.size, size=1)
        params = lstm.fU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.dfU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test prev state to output gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.oU.size, size=1)
        params = lstm.oU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.doU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test prev state to input gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.iU.size, size=1)
        params = lstm.iU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.diU.flat[idx]
        assert np.allclose(analytic, numerical)


def test_cell_input_weight_gradients(lstm, snli):
    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    sample = random.choice(snli.data)
    xs = random.choice(sample)
    ys = np.zeros(n_labels)
    ys[np.random.randint(0, n_labels, 1)] = 1
    ys = ys.reshape(n_labels, 1)

    # test prev state to cell input weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.uU.size, size=1)
        params = lstm.uU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.duU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to cell input weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.uW.size, size=1)
        params = lstm.uW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.duW.flat[idx]
        assert np.allclose(analytic, numerical)


def test_bias_gradients(lstm, snli):
    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    sample = random.choice(snli.data)
    xs = random.choice(sample)
    ys = np.zeros(n_labels)
    ys[np.random.randint(0, n_labels, 1)] = 1
    ys = ys.reshape(n_labels, 1)

    # test bias gradient for input gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.i_bias.size, size=1)
        params = lstm.i_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.i_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for forget gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.f_bias.size, size=1)
        params = lstm.f_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.f_bias_grad.flat[idx]

    # test bias gradient for output gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.o_bias.size, size=1)
        params = lstm.o_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.o_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for cell input
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.u_bias.size, size=1)
        params = lstm.u_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, logreg)

        lstm.forward_pass(xs)

        logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(logreg.yi_grad, rate=0.001)
        analytic = lstm.u_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)


def test_input_vector_gradients(lstm, snli):
    dim = 50
    n_labels = 3
    n_gradient_checks = 25

    logreg = LogisticRegression(n_features=dim, n_labels=n_labels)

    sample = random.choice(snli.data)
    xs = random.choice(sample)
    ys = np.zeros(n_labels)
    ys[np.random.randint(0, n_labels, 1)] = 1
    ys = ys.reshape(n_labels, 1)

    lstm.forward_pass(xs)
    words = [word for word in lstm.sen]

    for _ in range(n_gradient_checks):
        for word in words:
            lcase = word.lower()
            idx = np.random.randint(0, lstm.vectors[lcase].size, size=1)
            params = lstm.vectors[lcase].flat

            numerical = num_grad(lstm, params, idx, xs, ys, logreg)

            lstm.forward_pass(xs)

            logreg.train(lstm.get_root_embedding(), ys, rate=0.001)

            lstm.backward_pass(logreg.yi_grad, rate=0.001)

            analytic = lstm.dxs[lcase].flat[idx]
            assert np.allclose(analytic, numerical)
