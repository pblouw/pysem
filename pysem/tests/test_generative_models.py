import numpy as np

from pysem.networks import DependencyNetwork


def test_embedding_generator(embedding_generator, snli):
    dim = 25
    iters = 50
    rate = 0.01
    data = [d for d in snli.test_data]

    encoder = DependencyNetwork(dim=dim, vocab=snli.vocab)

    for _ in range(iters):
        for sample in data:
            s1 = sample.sentence1
            s2 = sample.sentence2

            encoder.forward_pass(s1)
            embedding_generator.forward_pass(s2, encoder.get_root_embedding())
            embedding_generator.backward_pass(rate=rate)
            encoder.backward_pass(embedding_generator.pass_grad, rate=rate)

    for node in embedding_generator.tree:
        assert node.computed
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.tvals, np.ndarray)
        assert isinstance(node.pword, str)
        assert node.lower_ == node.pword
