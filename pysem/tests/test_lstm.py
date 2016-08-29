import random
import numpy as np


def test_forward_pass(lstm, snli):
    sample = next(snli.train_data)
    sen = random.choice(sample)

    lstm.forward_pass(sen)
    sen_vec = lstm.get_root_embedding()

    assert isinstance(sen_vec, np.ndarray)

    for node in lstm.tree:
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.cell_state, np.ndarray)
