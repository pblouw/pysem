import random
import numpy as np

n_gradient_checks = 25
n_labels = 3
rate = 0.001


def random_data(snli):
    sample = random.choice(snli.data)
    x = random.choice(sample)
    y = np.zeros((n_labels, 1))
    y[np.random.randint(0, n_labels, 1), 0] = 1

    return x, y


def train_step(s_model, classifier, xs, ys):
    s_model.forward_pass(xs)
    classifier.train(s_model.get_root_embedding(), ys, rate=rate)
    s_model.backward_pass(classifier.yi_grad, rate=rate)


def test_forward_pass(treeLSTM, snli):
    sample = next(snli.train_data)
    sen = random.choice(sample)

    treeLSTM.forward_pass(sen)

    assert isinstance(treeLSTM.get_root_embedding(), np.ndarray)

    for node in treeLSTM.tree:
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.cell_state, np.ndarray)
        assert isinstance(node.inp_vec, np.ndarray)
        assert isinstance(node.h_tilda, np.ndarray)
        assert isinstance(node.i_gate, np.ndarray)
        assert isinstance(node.o_gate, np.ndarray)
        assert isinstance(node.f_gates, dict)


def test_backward_pass(treeLSTM, snli):
    dim = 50
    eps = 0.5

    sample = next(snli.train_data)
    sen = random.choice(sample)

    error_grad = np.random.random((dim, 1)) * 2 * eps - eps

    treeLSTM.forward_pass(sen)
    treeLSTM.backward_pass(error_grad)

    for node in treeLSTM.tree:
        assert isinstance(node.h_grad, np.ndarray)


def test_gate_gradients(treeLSTM, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # test input to output gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.oW.size, size=1)
        params = treeLSTM.oW.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.doW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to forget gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.fW.size, size=1)
        params = treeLSTM.fW.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.dfW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to input gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.iW.size, size=1)
        params = treeLSTM.iW.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.diW.flat[idx]
        assert np.allclose(analytic, numerical)

    # test hidden to output gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.oU.size, size=1)
        params = treeLSTM.oU.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.doU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test hidden to forget gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.fU.size, size=1)
        params = treeLSTM.fU.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.dfU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test hidden to input gate weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.iU.size, size=1)
        params = treeLSTM.iU.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.diU.flat[idx]
        assert np.allclose(analytic, numerical)


def test_cell_inp_gradients(treeLSTM, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # test prev state to cell input weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.uU.size, size=1)
        params = treeLSTM.uU.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.duU.flat[idx]
        assert np.allclose(analytic, numerical)

    # test input to cell input weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.uW.size, size=1)
        params = treeLSTM.uW.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.duW.flat[idx]
        assert np.allclose(analytic, numerical)


def test_cell_out_gradients(treeLSTM, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # test prev state to cell input weights
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.yW.size, size=1)
        params = treeLSTM.yW.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.dyW.flat[idx]
        assert np.allclose(analytic, numerical)


def test_bias_gradients(treeLSTM, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    # test bias gradient for input gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.i_bias.size, size=1)
        params = treeLSTM.i_bias.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.i_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for forget gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.f_bias.size, size=1)
        params = treeLSTM.f_bias.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.f_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for output gate
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.o_bias.size, size=1)
        params = treeLSTM.o_bias.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.o_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for cell input
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.u_bias.size, size=1)
        params = treeLSTM.u_bias.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.u_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)

    # test bias gradient for cell output
    for _ in range(n_gradient_checks):
        idx = np.random.randint(0, treeLSTM.y_bias.size, size=1)
        params = treeLSTM.y_bias.flat

        numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
        train_step(treeLSTM, classifier, xs, ys)
        analytic = treeLSTM.y_bias_grad.flat[idx]
        assert np.allclose(analytic, numerical)


def test_input_vec_gradients(treeLSTM, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    treeLSTM.forward_pass(xs)
    words = [n.lower_ for n in treeLSTM.tree]

    # Use random element in each word embedding for n numerical gradient checks
    for _ in range(n_gradient_checks):
        for word in words:
            idx = np.random.randint(0, treeLSTM.vectors[word].size, size=1)
            params = treeLSTM.vectors[word].flat

            numerical = num_grad(treeLSTM, params, idx, xs, ys, classifier)
            train_step(treeLSTM, classifier, xs, ys)
            analytic = treeLSTM.x_grads[word].flat[idx]

            assert np.allclose(analytic, numerical)
