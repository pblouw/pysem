import random
import numpy as np


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
