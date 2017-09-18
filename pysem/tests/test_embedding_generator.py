import random
import numpy as np

from pysem.generatives import EncoderDecoder

n_gradient_checks = 25
rate = 0.01
delta = 1e-6


def num_grad(params, idx, embgen, dnn, s2):
    val = np.copy(params[idx])

    params[idx] = val + delta
    pcost = embgen.get_cost(s2, dnn.get_root_embedding())
    params[idx] = val - delta
    ncost = embgen.get_cost(s2, dnn.get_root_embedding())
    params[idx] = val
    numerical_grad = (pcost - ncost) / (2 * delta)

    return numerical_grad


def test_forward_pass(embgen, dnn, snli):
    sample = random.choice(snli.train_data)
    s1 = sample.sentence1
    s2 = sample.sentence2

    dnn.forward_pass(s1)
    embgen.forward_pass(s2, dnn.get_root_embedding())

    for node in embgen.tree:
        assert node.computed
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.tvals, np.ndarray)
        assert isinstance(node.pword, str)


def test_backward_pass(embgen, dnn, snli):
    sample = random.choice(snli.train_data)
    s1 = sample.sentence1
    s2 = sample.sentence2

    for _ in range(50):
        dnn.forward_pass(s1)
        embgen.forward_pass(s2, dnn.get_root_embedding())
        embgen.backward_pass(rate=rate)
        dnn.backward_pass(embgen.pass_grad, rate=rate)

    node = random.choice(embgen.tree)
    assert node.pword == node.lower_


def test_dep_weight_gradients(embgen, dnn, snli):
    sample = random.choice(snli.train_data)
    s1 = sample.sentence1
    s2 = sample.sentence2

    dnn.forward_pass(s1)
    embgen.forward_pass(s2, dnn.get_root_embedding())
    deps = [node.dep_ for node in dnn.tree if node.dep_ != 'ROOT']

    for dep in deps:
        for _ in range(n_gradient_checks):
            dnn.forward_pass(s1)

            idx = np.random.randint(0, embgen.d_weights[dep].size, size=1)
            params = embgen.d_weights[dep].flat

            numerical_grad = num_grad(params, idx, embgen, dnn, s2)

            embgen.forward_pass(s2, dnn.get_root_embedding())
            embgen.backward_pass(rate=rate)
            dnn.backward_pass(embgen.pass_grad, rate=rate)

            analytic_grad = embgen.dgrads[dep].flat[idx]

            assert np.allclose(analytic_grad, numerical_grad)


def test_word_weight_gradients(embgen, dnn, snli):
    sample = random.choice(snli.train_data)
    s1 = sample.sentence1
    s2 = sample.sentence2

    dnn.forward_pass(s1)
    embgen.forward_pass(s2, dnn.get_root_embedding())
    deps = [node.dep_ for node in dnn.tree if node.dep_ != 'ROOT']

    for dep in deps:
        for _ in range(n_gradient_checks):
            dnn.forward_pass(s1)

            idx = np.random.randint(0, embgen.w_weights[dep].size, size=1)
            params = embgen.w_weights[dep].flat

            numerical_grad = num_grad(params, idx, embgen, dnn, s2)

            embgen.forward_pass(s2, dnn.get_root_embedding())
            embgen.backward_pass(rate=rate)
            dnn.backward_pass(embgen.pass_grad, rate=rate)

            analytic_grad = embgen.wgrads[dep].flat[idx]

            assert np.allclose(analytic_grad, numerical_grad)


def test_encoder_decoder(embgen, dnn, snli):
    sample = random.choice(snli.train_data)
    s1 = sample.sentence1
    s2 = sample.sentence2

    model = EncoderDecoder(dnn, embgen, snli.train_data)
    model.encode(s1)
    probs = model.decode(s2, n_probs=1)

    assert len(probs) == 1

    model.train(iters=1, rate=rate, batchsize=1)

    assert all(n.computed for n in model.encoder.tree)
    assert all(n.computed for n in model.decoder.tree)
