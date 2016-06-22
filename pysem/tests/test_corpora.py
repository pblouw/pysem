import os
import re
import random
import pytest
import types
import itertools

from pysem.corpora import Wikipedia, SNLI
from pysem.mputils import max_strip

wiki_path = os.getcwd() + '/pysem/tests/corpora/wikipedia/'
snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'


def test_wiki_streaming():
    wp = Wikipedia(wiki_path)

    assert isinstance(wp.articles, itertools.islice)
    assert isinstance(wp.sentences, itertools.islice)

    assert isinstance(next(wp.articles), str)
    assert isinstance(next(wp.sentences), str)

    all_articles = [a for a in wp.articles]
    assert len(all_articles) < 100


def test_wiki_preprocessing():
    wp = Wikipedia(wiki_path)
    article = next(wp.articles)

    pattern = re.compile(r"<.*>")
    assert not pattern.findall(article)

    pattern = re.compile(r"\n")
    assert not pattern.findall(article)


def test_wiki_caching(tmpdir):
    cache_path = str(tmpdir) + '/'

    wp = Wikipedia(wiki_path)
    wp.write_to_cache(cache_path, process=max_strip, batchsize=100)

    wp_cache = Wikipedia(cache_path, from_cache=True)
    article = next(wp_cache.articles)
    assert isinstance(article, str)

    pattern = re.compile(r"[0-9]|\[|\<|\(|\"|\'|\{")
    assert not pattern.findall(article)

    all_cached_articles = [a for a in wp_cache.articles]
    assert len(all_cached_articles) < 100


def test_wiki_vocab_build():
    wp = Wikipedia(wiki_path, article_limit=1)
    wp.build_vocab(threshold=0.05, batchsize=1)

    assert isinstance(wp.vocab, list)
    assert isinstance(random.choice(wp.vocab), str)
    assert len(wp.vocab) > 100

    assert isinstance(next(wp.articles), str)


def test_wiki_stream_reset():
    wp = Wikipedia(wiki_path)
    for _ in wp.articles:
        continue

    with pytest.raises(StopIteration):
        next(wp.articles)

    wp.reset_streams()
    assert isinstance(next(wp.articles), str)


def test_wiki_article_limit():
    wp = Wikipedia(wiki_path, article_limit=1)
    all_articles = [a for a in wp.articles]

    assert len(all_articles) == wp.article_limit


def test_snli_streaming():
    snli = SNLI(snli_path)

    assert isinstance(snli.dev_data, types.GeneratorType)
    assert isinstance(snli.test_data, types.GeneratorType)
    assert isinstance(snli.train_data, types.GeneratorType)

    assert isinstance(next(snli.dev_data), dict)
    assert isinstance(next(snli.test_data), dict)
    assert isinstance(next(snli.train_data), dict)

    samples = [d for d in snli.dev_data]
    assert len(samples) < 40


def test_snli_vocab_build():
    snli = SNLI(snli_path)
    snli.build_vocab()

    assert isinstance(snli.vocab, list)
    assert isinstance(random.choice(snli.vocab), str)
    assert len(snli.vocab) > 100
