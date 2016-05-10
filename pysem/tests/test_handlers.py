import os
import re
import types

from pysem.handlers import Wikipedia

corpus_path = os.getcwd() + '/pysem/tests/corpora'


def test_handler():
    wp = Wikipedia(corpus_path)

    assert isinstance(wp.batches, types.GeneratorType)
    assert isinstance(wp.documents, types.GeneratorType)
    assert isinstance(wp.sentences, types.GeneratorType)

    assert isinstance(next(wp.batches), list)
    assert isinstance(next(wp.documents), str)
    assert isinstance(next(wp.sentences), str)


def test_preprocess():
    wp = Wikipedia(corpus_path)
    document = next(wp.documents)

    pattern = re.compile(r"<.*>")
    assert not pattern.findall(document)

    pattern = re.compile(r"\n")
    assert not pattern.findall(document)
