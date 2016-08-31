import random
import numpy as np


def test_forward_pass(treeLSTM, snli):
    sample = next(snli.train_data)
    sen = random.choice(sample)

    treeLSTM.forward_pass(sen)
    sen_vec = treeLSTM.get_root_embedding()

    assert isinstance(sen_vec, np.ndarray)

    for node in treeLSTM.tree:
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.cell_state, np.ndarray)


def test_backward_pass(treeLSTM, snli):
    dim = 50
    eps = 0.5

    sample = next(snli.train_data)
    sen = random.choice(sample)

    error_grad = np.random.random((dim, 1)) * 2 * eps - eps

    treeLSTM.forward_pass(sen)
    treeLSTM.backward_pass(error_grad)

    for node in treeLSTM.tree:
        assert isinstance(node.top_grad, np.ndarray)
