import os
import re
import random
import pytest
import itertools

from pysem.handlers import Wikipedia
from pysem.mputils import max_strip

corpus_path = os.getcwd() + '/pysem/tests/corpora/'


def test_streaming():
    wp = Wikipedia(corpus_path)

    assert isinstance(wp.articles, itertools.islice)
    assert isinstance(wp.sentences, itertools.islice)

    assert isinstance(next(wp.articles), str)
    assert isinstance(next(wp.sentences), str)

    all_articles = [a for a in wp.articles]
    assert len(all_articles) < 100


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
    wp.write_to_cache(cache_path, process=max_strip, n_per_file=100)

    wp_cache = Wikipedia(cache_path, from_cache=True)
    article = next(wp_cache.articles)
    assert isinstance(article, str)

    pattern = re.compile(r"[0-9]|\[|\<|\(|\"|\'|\{")
    assert not pattern.findall(article)

    all_cached_articles = [a for a in wp_cache.articles]
    assert len(all_cached_articles) < 100


def test_vocab_build():
    wp = Wikipedia(corpus_path, article_limit=1)
    wp.build_vocab(cutoff=0.05, batchsize=1)

    assert isinstance(wp.vocab, list)
    assert isinstance(random.choice(wp.vocab), str)
    assert len(wp.vocab) > 100

    assert isinstance(next(wp.articles), str)


def test_stream_reset():
    wp = Wikipedia(corpus_path)
    for _ in wp.articles:
        continue

    with pytest.raises(StopIteration):
        next(wp.articles)

    wp.reset()
    assert isinstance(next(wp.articles), str)


def test_article_limit():
    wp = Wikipedia(corpus_path, article_limit=1)
    all_articles = [a for a in wp.articles]

    assert len(all_articles) == wp.article_limit
