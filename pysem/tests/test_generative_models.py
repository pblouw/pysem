import os
import spacy

import numpy as np

from pysem.corpora import SNLI
from pysem.networks import DependencyNetwork
from pysem.generatives import NodeGenerator

snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'
parser = spacy.load('en')


def test_node_generator():
    snli = SNLI(snli_path)
    snli.build_vocab()
    snli.extractor = snli.get_sentences

    data = [d for d in snli.train_data]

    # build subvocabs for each dependency
    subvocabs = {dep: set() for dep in DependencyNetwork.deps}

    for sample in data:
        s1_parse = parser(sample.sentence1)
        s2_parse = parser(sample.sentence2)

        for token in s1_parse:
            if token.lower_ not in subvocabs[token.dep_]:
                subvocabs[token.dep_].add(token.lower_)

        for token in s2_parse:
            if token.lower_ not in subvocabs[token.dep_]:
                subvocabs[token.dep_].add(token.lower_)

    dim = 100
    iters = 50
    rate = 0.005

    encoder = DependencyNetwork(dim=dim, vocab=snli.vocab)
    decoder = NodeGenerator(dim=dim, subvocabs=subvocabs)

    for _ in range(iters):
        for sample in data:
            s1 = sample.sentence1
            s2 = sample.sentence2

            encoder.forward_pass(s1)
            decoder.forward_pass(s2, encoder.get_root_embedding())
            decoder.backward_pass(rate=rate)
            encoder.backward_pass(decoder.pass_grad, rate=rate)

    for node in decoder.tree:
        assert node.computed
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.tvals, np.ndarray)
        assert isinstance(node.pword, str)
        assert node.lower_ == node.pword
