import os
import re
import random
import types
import pytest

from pysem.handlers import Wikipedia
from pysem.mputils import max_strip

corpus_path = os.getcwd() + '/pysem/tests/corpora/'


def test_streaming():
    wp = Wikipedia(corpus_path)

    assert isinstance(wp.batches, types.GeneratorType)
    assert isinstance(wp.articles, types.GeneratorType)
    assert isinstance(wp.sentences, types.GeneratorType)

    assert isinstance(next(wp.batches), list)
    assert isinstance(next(wp.articles), str)
    assert isinstance(next(wp.sentences), str)


def test_preprocessing():
    wp = Wikipedia(corpus_path)
    article = next(wp.articles)

    pattern = re.compile(r"<.*>")
    assert not pattern.findall(article)

    pattern = re.compile(r"\n")
    assert not pattern.findall(article)


def test_caching(tmpdir):
    cache_path = str(tmpdir) + '/'

    wp = Wikipedia(corpus_path)
    wp.write_to_cache(cache_path, processor=max_strip, n_per_file=1)

    wp_cache = Wikipedia(cache_path, from_cache=True)
    article = next(wp_cache.articles)
    assert isinstance(article, str)

    pattern = re.compile(r"[0-9]|\[|\<|\(|\"|\'|\{")
    assert not pattern.findall(article)

    all_cached_articles = [a for a in wp_cache.articles]
    assert len(all_cached_articles) < 90


def test_vocab_build():
    wp = Wikipedia(corpus_path)
    wp.build_vocab(cutoff=0.05, batchsize=1)

    assert isinstance(wp.vocab, list)
    assert isinstance(random.choice(wp.vocab), str)
    assert len(wp.vocab) > 1000

    assert isinstance(next(wp.articles), str)


def test_stream_reset():
    wp = Wikipedia(corpus_path)
    for _ in wp.articles:
        continue

    with pytest.raises(StopIteration):
        next(wp.articles)

    wp._reset_streams()
    assert isinstance(next(wp.articles), str)
