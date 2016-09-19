import os
import re
import random
import pytest
import itertools

import numpy as np

from pysem.corpora import Wikipedia, SNLI, SICK
from pysem.utils.multiprocessing import max_strip

sick_path = os.getcwd() + '/pysem/tests/corpora/sick/sicktest.txt'


def test_wiki_streaming(wikipedia):
    wikipedia.reset_streams()

    assert isinstance(wikipedia.articles, itertools.islice)
    assert isinstance(wikipedia.sentences, itertools.islice)

    assert isinstance(next(wikipedia.articles), str)
    assert isinstance(next(wikipedia.sentences), str)

    all_articles = [a for a in wikipedia.articles]
    assert len(all_articles) < 100


def test_wiki_preprocessing(wikipedia):
    wikipedia.reset_streams()

    article = next(wikipedia.articles)

    pattern = re.compile(r"<.*>")
    assert not pattern.findall(article)

    pattern = re.compile(r"\n")
    assert not pattern.findall(article)


def test_wiki_caching(tmpdir, wikipedia):
    wikipedia.reset_streams()
    cache_path = str(tmpdir) + '/'
    wikipedia.write_to_cache(cache_path, process=max_strip, batchsize=100)

    wikipedia_cache = Wikipedia(cache_path, from_cache=True)
    article = next(wikipedia_cache.articles)
    assert isinstance(article, str)

    pattern = re.compile(r"[0-9]|\[|\<|\(|\"|\'|\{")
    assert not pattern.findall(article)

    all_cached_articles = [a for a in wikipedia_cache.articles]
    assert len(all_cached_articles) < 100


def test_wiki_vocab_build(tmpdir, wikipedia):
    wikipedia.build_vocab(threshold=0.05, batchsize=1)

    assert isinstance(wikipedia.vocab, list)
    assert isinstance(random.choice(wikipedia.vocab), str)
    assert len(wikipedia.vocab) > 50

    vocab_path = str(tmpdir) + '/'
    wikipedia.save_vocab(vocab_path + 'test.pickle')

    wikipedia.vocab = None
    wikipedia.load_vocab(vocab_path + 'test.pickle')

    assert isinstance(wikipedia.vocab, list)
    assert isinstance(random.choice(wikipedia.vocab), str)
    assert len(wikipedia.vocab) > 50


def test_wiki_stream_reset(wikipedia):
    wikipedia.reset_streams()
    for _ in wikipedia.articles:
        continue

    with pytest.raises(StopIteration):
        next(wikipedia.articles)

    wikipedia.reset_streams()
    assert isinstance(next(wikipedia.articles), str)


def test_wiki_article_limit(wikipedia):
    wikipedia.reset_streams()
    all_articles = [a for a in wikipedia.articles]

    assert len(all_articles) == wikipedia.article_limit


def test_snli_data(snli):
    snli.load_raw()

    assert isinstance(snli.dev_data, list)
    assert isinstance(snli.test_data, list)
    assert isinstance(snli.train_data, list)
    assert len(snli.dev_data) < 40


def test_snli_vocab_build(snli):
    snli.build_vocab()

    assert isinstance(snli.vocab, list)
    assert isinstance(random.choice(snli.vocab), str)
    assert len(snli.vocab) > 100


def test_snli_extractors(snli):
    snli.load_xy_pairs()
    xy_pair = random.choice(snli.test_data)

    assert isinstance(xy_pair, tuple)
    assert isinstance(random.choice(xy_pair), str)

    snli.load_sentences()
    sen_pair = random.choice(snli.test_data)

    assert isinstance(sen_pair, tuple)
    assert isinstance(random.choice(sen_pair), str)

    snli.load_text()

    assert isinstance(snli.test_data, str)

    snli.load_parses()
    parse_pair = random.choice(snli.test_data)

    assert isinstance(parse_pair, tuple)

    snli.load_binary_parses()
    parse_pair = random.choice(snli.test_data)

    assert isinstance(parse_pair, tuple)


def test_snli_binarizer():
    labels = ['entailment', 'neutral', 'contradiction']
    array = SNLI.binarize(labels)

    assert np.sum(array) == len(labels)
    assert array[0, 0] == 1
    assert array[0, 1] == 0
    assert array[1, 1] == 1
    assert array[1, 0] == 0
    assert array[2, 2] == 1


def test_sick_streaming():
    sick = SICK(sick_path)

    assert isinstance(sick.entailment_pairs, list)
    assert isinstance(sick.relatedness_pairs, list)
    assert isinstance(sick.data, list)

    entailment_sample = random.choice(sick.entailment_pairs)
    relatedness_sample = random.choice(sick.relatedness_pairs)

    assert isinstance(entailment_sample.sentence1, str)
    assert isinstance(entailment_sample.sentence2, str)
    assert isinstance(entailment_sample.label, str)

    assert isinstance(relatedness_sample.sentence1, str)
    assert isinstance(relatedness_sample.sentence2, str)
    assert isinstance(relatedness_sample.score, str)


def test_sick_vocab_build():
    sick = SICK(sick_path)
    sick.build_vocab()

    assert isinstance(sick.vocab, list)
    assert isinstance(random.choice(sick.vocab), str)
    assert len(sick.vocab) > 100
