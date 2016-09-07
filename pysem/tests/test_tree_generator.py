import random
import numpy as np


def test_tree_generator(tree_gen, dnn, snli):
    iters = 50
    rate = 0.1

    sample = random.choice(snli.data)
    s1 = sample.sentence1
    s2 = sample.sentence2

    for _ in range(iters):
        dnn.forward_pass(s1)
        tree_gen.forward_pass(dnn.get_root_embedding(), s2)
        tree_gen.backward_pass(rate=rate)
        dnn.backward_pass(tree_gen.pass_grad, rate=rate)

    for node in tree_gen.sequence:
        print(node.lower_)
        assert node.computed
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.py_w, np.ndarray)
        assert isinstance(node.py_h, np.ndarray)
        assert isinstance(node.py_d, np.ndarray)

        assert isinstance(node.pw, str)
        assert isinstance(node.ph, str)
        assert isinstance(node.pd, str)

        assert node.lower_ == node.pw
        assert node.dep_ == node.pd
        assert node.head.lower_ == node.ph
