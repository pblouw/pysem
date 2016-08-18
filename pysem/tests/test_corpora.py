import os
import re
import random
import pytest
import types
import itertools

import numpy as np

from pysem.corpora import Wikipedia, SNLI, SICK
from pysem.utils.multiprocessing import max_strip

wiki_path = os.getcwd() + '/pysem/tests/corpora/wikipedia/'
snli_path = os.getcwd() + '/pysem/tests/corpora/snli/'
sick_path = os.getcwd() + '/pysem/tests/corpora/sick/sicktest.txt'


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


def test_wiki_vocab_build(tmpdir):
    wp = Wikipedia(wiki_path, article_limit=1)
    wp.build_vocab(threshold=0.05, batchsize=1)

    assert isinstance(wp.vocab, list)
    assert isinstance(random.choice(wp.vocab), str)
    assert len(wp.vocab) > 50

    vocab_path = str(tmpdir) + '/'
    wp.save_vocab(vocab_path + 'test.pickle')

    wp.vocab = None
    wp.load_vocab(vocab_path + 'test.pickle')

    assert isinstance(wp.vocab, list)
    assert isinstance(random.choice(wp.vocab), str)
    assert len(wp.vocab) > 50


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


def test_snli_extractors():
    snli = SNLI(snli_path)
    snli.extractor = snli.get_xy_pairs

    xy_pair = next(snli.dev_data)

    assert isinstance(xy_pair, tuple)
    assert isinstance(random.choice(xy_pair), str)

    snli.extractor = snli.get_sentences
    sentences = next(snli.dev_data)

    assert isinstance(sentences, tuple)
    assert isinstance(random.choice(sentences), str)

    snli.extractor = snli.get_text
    test_text = snli.test_data

    assert isinstance(test_text, str)

    snli.extractor = snli.get_parses
    parses = next(snli.train_data)

    assert isinstance(parses, tuple)

    snli.extractor = snli.get_binary_parses
    parses = next(snli.train_data)

    assert isinstance(parses, tuple)


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
