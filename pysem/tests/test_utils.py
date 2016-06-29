import re
import os
import random
import collections

from pysem.corpora import Wikipedia
from pysem.utils.multiprocessing import max_strip, basic_strip, count_words
from pysem.utils.multiprocessing import flatten


wiki_path = os.getcwd() + '/pysem/tests/corpora/wikipedia/'


def test_max_strip():
    wp = Wikipedia(wiki_path)
    article = max_strip(next(wp.articles))

    assert isinstance(article, str)

    pattern = re.compile(r"[0-9]|\[|\<|\(|\"|\'|\{")
    assert not pattern.findall(article)


def test_basic_strip():
    wp = Wikipedia(wiki_path)
    article = basic_strip(next(wp.articles))

    assert isinstance(article, str)


def test_count_words():
    wp = Wikipedia(wiki_path)
    article = next(wp.articles)

    counts = count_words(article)
    article = basic_strip(article)

    assert isinstance(counts, collections.Counter)

    ntokens = sum([c for c in counts.values()])
    wordset = article.split()

    assert len(wordset) - ntokens < 500

def test_flatten():
    n_nested = 10
    nest_size = 5 
    testlist = [list(range(nest_size)) for _ in range(n_nested)]
    flatlist = flatten(testlist)

    assert len(flatlist) == n_nested * nest_size 
