import re
import pytest
import collections
import numpy as np

from pysem.utils.multiprocessing import max_strip, basic_strip, count_words
from pysem.utils.multiprocessing import flatten
from pysem.utils.vsa import HRR


def test_max_strip(wikipedia):
    wikipedia.reset_streams()
    article = max_strip(next(wikipedia.articles))

    assert isinstance(article, str)

    pattern = re.compile(r"[0-9]|\[|\<|\(|\"|\'|\{")
    assert not pattern.findall(article)


def test_basic_strip(wikipedia):
    wikipedia.reset_streams()
    article = basic_strip(next(wikipedia.articles))

    assert isinstance(article, str)


def test_count_words(wikipedia):
    wikipedia.reset_streams()
    article = next(wikipedia.articles)

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


def test_hrr():
    dim = 100
    hrr_a = HRR(np.random.normal(0, 1/dim**0.5, dim))
    hrr_b = HRR(np.zeros(dim))
    hrr_b.v[0] = 1

    assert isinstance(hrr_a + hrr_b, HRR)
    assert isinstance(hrr_a * hrr_b, HRR)
    assert isinstance(hrr_a - hrr_b, HRR)

    prod = hrr_a * hrr_b
    assert np.allclose(prod.v, hrr_a.v)

    inv = ~hrr_b
    assert np.allclose(inv.v, hrr_b.v)

    prodinv = hrr_b * ~hrr_b
    assert np.allclose(prodinv.v, hrr_b.v)

    hrr_c = HRR(np.random.normal(0, 1/dim**0.5, dim), unitary=True)
    hrr_b.make_unitary()

    unitary_unbind = (hrr_b * hrr_c) * ~hrr_b
    assert np.allclose(hrr_c.v, unitary_unbind.v)

    assert hrr_c.norm_of_vector() != 1.0
    hrr_c.normalize()
    assert hrr_c.norm_of_vector() == 1.0

    with pytest.raises(Exception):
        hrr_a * 5
        hrr_b - 5
        hrr_c + 5
