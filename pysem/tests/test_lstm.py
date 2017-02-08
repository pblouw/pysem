import random
import numpy as np

n_gradient_checks = 25
n_labels = 3
bsize = 10
rate = 0.001


def random_data(snli):
    samples = [random.choice(snli.train_data) for _ in range(bsize)]
    xs = [random.choice(sample) for sample in samples]
    ys = np.zeros((n_labels, bsize))
    ys[np.random.randint(0, n_labels, bsize), list(range(bsize))] = 1

    return xs, ys


def train_step(s_model, classifier, xs, ys):
    s_model.forward_pass(xs)
    classifier.train(s_model.get_root_embedding(), ys, rate=rate)
    s_model.backward_pass(classifier.yi_grad, rate=rate)


def test_forward_pass(lstm, snli):
    xs, ys = random_data(snli)

    lstm.forward_pass(xs)
    assert isinstance(lstm.get_root_embedding(), np.ndarray)

    for i in range(lstm.seqlen):
        assert isinstance(lstm.i_gates[i], np.ndarray)
        assert isinstance(lstm.f_gates[i], np.ndarray)
        assert isinstance(lstm.o_gates[i], np.ndarray)
        assert isinstance(lstm.cell_inputs[i], np.ndarray)
        assert isinstance(lstm.cell_states[i], np.ndarray)
        assert isinstance(lstm.hs[i], np.ndarray)

        assert lstm.i_gates[0].shape == (lstm.c_dim, bsize)
        assert lstm.f_gates[0].shape == (lstm.c_dim, bsize)
        assert lstm.o_gates[0].shape == (lstm.c_dim, bsize)
        assert lstm.cell_inputs[0].shape == (lstm.c_dim, bsize)
        assert lstm.cell_states[0].shape == (lstm.c_dim, bsize)
        assert lstm.hs[0].shape == (lstm.c_dim, bsize)
        assert lstm.ys.shape == (lstm.i_dim, bsize)


def test_gate_weight_gradients(lstm, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # test input to forget gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.oW.size, size=1)
        params = lstm.oW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.doW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to output gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.fW.size, size=1)
        params = lstm.fW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.dfW.flat[idx]

        assert np.allclose(analytic, numerical)

    # test input to input gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.iW.size, size=1)
        params = lstm.iW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.diW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test prev state to forget gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.fU.size, size=1)
        params = lstm.fU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.dfU.flat[idx]
        assert np.allclose(analytic, numerical)

    # # test prev state to output gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.oU.size, size=1)
        params = lstm.oU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)

        lstm.forward_pass(xs)

        classifier.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(classifier.yi_grad, rate=0.001)
        analytic = lstm.doU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test prev state to input gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.iU.size, size=1)
        params = lstm.iU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)

        lstm.forward_pass(xs)

        classifier.train(lstm.get_root_embedding(), ys, rate=0.001)

        lstm.backward_pass(classifier.yi_grad, rate=0.001)
        analytic = lstm.diU.flat[idx]
        assert np.allclose(analytic, numerical)


def test_cell_input_gradients(lstm, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # test prev state to cell input weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.uU.size, size=1)
        params = lstm.uU.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.duU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to cell input weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.uW.size, size=1)
        params = lstm.uW.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.duW.flat[idx]
        assert np.allclose(analytic, numerical)


def test_bias_gradients(lstm, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # test bias gradient for input gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.i_bias.size, size=1)
        params = lstm.i_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.i_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for forget gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.f_bias.size, size=1)
        params = lstm.f_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.f_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for output gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.o_bias.size, size=1)
        params = lstm.o_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.o_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for cell input
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, lstm.u_bias.size, size=1)
        params = lstm.u_bias.flat

        numerical = num_grad(lstm, params, idx, xs, ys, classifier)
        train_step(lstm, classifier, xs, ys)
        analytic = lstm.u_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)


def test_input_vector_gradients(lstm, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    lstm.forward_pass(xs)
    words = lstm.batch[0]
    words = [w for w in words if w != 'PAD']
    words = [w for w in words if w in snli.vocab]

    # Use random element in each word embedding for n numerical gradient checks
    for _ in range(n_gradient_checks):
        for word in words:
            idx = np.random.randint(0, lstm.vectors[word].size, size=1)
            params = lstm.vectors[word].flat

            numerical_grad = num_grad(lstm, params, idx, xs, ys, classifier)
            train_step(lstm, classifier, xs, ys)
            analytic_grad = lstm.x_grads[word].flat[idx]

            assert np.allclose(analytic_grad, numerical_grad)
