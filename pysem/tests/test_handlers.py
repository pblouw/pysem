import os
import re
import types

from pysem.handlers import Wikipedia

corpus_path = os.getcwd() + '/pysem/tests/corpora'


def test_handler():
    wp = Wikipedia(corpus_path)

    assert isinstance(wp.batches, types.GeneratorType)
    assert isinstance(wp.articles, types.GeneratorType)
    assert isinstance(wp.sentences, types.GeneratorType)

    assert isinstance(next(wp.batches), list)
    assert isinstance(next(wp.articles), str)
    assert isinstance(next(wp.sentences), str)


def test_preprocess():
    wp = Wikipedia(corpus_path)
    article = next(wp.articles)

    pattern = re.compile(r"<.*>")
    assert not pattern.findall(article)

    pattern = re.compile(r"\n")
    assert not pattern.findall(article)
