import random
import pytest
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
