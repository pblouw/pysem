import random
import numpy as np


def test_tree_generator(tree_gen, dnn, snli):
    iters = 100
    rate = 0.1
    n = 10

    sample = random.choice(snli.train_data)
    s1 = sample.sentence1
    s2 = sample.sentence2

    for _ in range(iters):
        dnn.forward_pass(s1)
        tree_gen.forward_pass(dnn.get_root_embedding(), s2)
        tree_gen.backward_pass(rate=rate)
        dnn.backward_pass(tree_gen.pass_grad, rate=rate)

    for node in tree_gen.sequence:

        assert node.computed
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.py_w, np.ndarray)
        assert isinstance(node.py_h, np.ndarray)
        assert isinstance(node.py_d, np.ndarray)

        assert isinstance(node.pw, str)
        assert isinstance(node.ph, str)
        assert isinstance(node.pd, str)

    tree_gen.predict(dnn.get_root_embedding(), n)

    assert len(tree_gen.sequence) == n
