import random
import numpy as np

n_labels = 3
n_gradient_checks = 25
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


def test_forward_pass(hnn, snli):
    sample = random.choice(snli.train_data)
    sen = random.choice(sample)

    hnn.forward_pass(sen)
    assert isinstance(hnn.get_root_embedding(), np.ndarray)

    for node in hnn.tree:
        assert isinstance(node.embedding, np.ndarray)


def test_backward_pass(hnn, snli):
    sample = random.choice(snli.train_data)
    sen = random.choice(sample)

    error_grad = np.random.random((hnn.dim, 1)) * 2 * 0.1 - 0.1

    hnn.forward_pass(sen)
    words = [node.lower_ for node in hnn.tree if node.lower_ in snli.vocab]

    # check that all of the words are in the vocab
    for word in words:
        assert word in snli.vocab

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
        assert np.linalg.norm(hnn.xgrads[word]) != 0

    # Save a copy of the word vectors after SGD update
    new_vectors = []
    for word in words:
        new_vectors.append(np.copy(hnn.vectors[word]))

    # Check that every word vector has changed after the SGD update
    for pair in zip(vectors, new_vectors):
        assert np.count_nonzero(pair[1] - pair[0]) == pair[0].size


def test_embedding_gradients(hnn, snli, get_cost, num_grad, classifier):
    xs, ys = random_data(snli)

    hnn.forward_pass(xs)
    words = [node.lower_ for node in hnn.tree if node.lower_ in snli.vocab]

    print(xs)
    for _ in range(n_gradient_checks):
        for word in words:
            print(word)
            idx = np.random.randint(0, hnn.vectors[word].size, size=1)
            params = hnn.vectors[word].flat

            numerical = num_grad(hnn, params, idx, xs, ys, classifier)
            hnn.forward_pass(xs)
            classifier.train(hnn.get_root_embedding(), ys, rate=0.001)
            hnn.backward_pass(classifier.yi_grad, rate=0.001)
            analytic = hnn.xgrads[word].flat[idx]

            assert np.allclose(analytic, numerical)
