import os
import re
import json
import sys
import nltk
import pickle
import itertools
import csv

import numpy as np

from collections import Counter, namedtuple
from itertools import islice
from pysem.utils.multiprocessing import plainmap, starmap, count_words

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class DataHandler(object):
    """A base class for handling datasets used in NLP applications"""
    def save_vocab(self, filename):
        '''Save a vocabulary to avoid recomputing word counts'''
        with open(filename, 'wb') as pfile:
            pickle.dump(self.vocab, pfile)

    def load_vocab(self, filename):
        '''Load a vocabulary that has been previously saved'''
        with open(filename, 'rb') as pfile:
            self.vocab = pickle.load(pfile)


class Wikipedia(DataHandler):
    """ A streaming interface to a dump of Wikipedia files. The dump files
    must be minimally preprocessed to remove html and other forms of
    markup.

    Parameters:
    ----------
    path : str
        The absolute path to the directory containing the dump files.
    article_limit : int
        The maximum number of articles to stream from the dump.
    from_cache : bool
        Flag to handle cached articles that have previously been processed.

    Attributes:
    ----------
    articles : generator
        Yields one article at a time until article limit is reached.
    sentences : generator
        Yields one sentence at a time until article limit is reached.
    """
    def __init__(self, path, article_limit=None, from_cache=False):
        self.path = path
        self.article_limit = article_limit if article_limit else sys.maxsize
        self.from_cache = from_cache
        self.reset_streams()

    @property
    def articles(self):
        return self._articles

    @articles.setter
    def articles(self, dummy):
        raise Exception('Articles are readonly and cannot be modified')

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, dummy):
        raise Exception('Sentences are readonly and cannot be modified')

    @staticmethod
    def preprocess(article):
        '''Perform basic preprocessing on a Wikipedia article'''
        article = re.sub("<.*>|<|>", "", article)
        article = article.split('\n')[3:]  # removes title from 1st sentence
        article = ' '.join(article)
        return article

    @staticmethod
    def cache(articles, process, path):
        '''Caches list of modified Wikipedia articles to specified path'''
        with open(path, 'w') as cachefile:
            for article in articles:
                cachefile.write(process(article) + '</doc>')

    def build_vocab(self, threshold=0.5, batchsize=100):
        '''Build a vocabularly of words from word frequency counts'''
        counter = Counter()
        while True:
            batch = list(islice(self.articles, batchsize))
            for counts in plainmap(count_words, batch):
                counter.update(counts)
            if len(batch) < batchsize:
                break

        self.vocab = counter.most_common(int(len(counter)*threshold))
        self.vocab = sorted([pair[0] for pair in self.vocab])
        self.reset_streams()

    def reset_streams(self):
        '''Reset all generators that stream data from the Wikipedia dump'''
        self._articles = islice(self._article_stream(), self.article_limit)
        self._sentences = islice(self._sentence_stream(), self.article_limit)

    def write_to_cache(self, path, process, batchsize=200, poolsize=10):
        '''Writes processed articles to cache in parallel for later use'''
        arglist = []
        for count in itertools.count(0):
            batch = list(islice(self.articles, batchsize))
            fname = str(count) + '.txt'
            arglist.append((batch, process, path + fname))

            if count % poolsize == 0 and count != 0:
                starmap(self.cache, arglist)
                arglist = []

            elif len(batch) < batchsize:
                starmap(self.cache, arglist)
                break

        self.reset_streams()

    def _article_stream(self):
        for root, dirs, files in os.walk(self.path):
            for fname in files:
                fpath = root + '/' + fname
                with open(fpath, 'r', encoding='ascii', errors='ignore') as f:
                    articles = f.read().split('</doc>')
                    for a in articles[1:]:
                        yield a if self.from_cache else self.preprocess(a)

    def _sentence_stream(self):
        for article in self._article_stream():
            sents = tokenizer.tokenize(article)
            sents = [s.replace('\n', '') for s in sents]
            for s in sents:
                yield s


Sentences = namedtuple('Sentences', ['sentence1', 'sentence2'])
Parses = namedtuple('Parses', ['parse1', 'parse2'])
BinaryParses = namedtuple('BinaryParses', ['parse1', 'parse2'])
TrainingPair = namedtuple('TrainingPair', ['sentence1', 'sentence2', 'label'])
RelationPair = namedtuple('RelationPair', ['sentence1', 'sentence2', 'score'])


class SNLI(DataHandler):
    """A streaming iterface to the SNLI corpus for natural language inference.
    The corpus data is provided as a json file, and this interface provides a
    number of methods for easily extracting particular subsets of the data.

    Each portion of the dataset is given its own stream, and setting an
    'extractor' modifies all streams in place to yield some more specific form
    of data (e.g. x-y pairs, parses, sentences, etc.)

    Parameters:
    ----------
    path : str
        The absolute path to the directory containing the SNLI corpus.

    Attributes:
    ----------
    dev_data : generator
        Yields one sample at a time from the development set.
    test_data : generator
        Yields one sample at a time from the test set.
    train_data : generator
        Yield one sample at a time from training set.
    extractor : function
        Decorates datastreams to yield more specific outputs. Initialized as
        the identity function, leaving the datastreams unchanged.
    """
    def __init__(self, path):
        self.path = path
        self.load_raw()

    @property
    def dev_data(self):
        return self._dev_data

    @dev_data.setter
    def dev_data(self, dummy):
        raise Exception('SNLI dev data is readonly and cannot be modified')

    @property
    def test_data(self):
        return self._test_data

    @test_data.setter
    def test_data(self, dummy):
        raise Exception('SNLI test data is readonly and cannot be modified')

    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, dummy):
        raise Exception('SNLI train data is readonly and cannot be modified')

    @staticmethod
    def binarize(labels):
        '''Turn a list of snli labels into a binary array to use in models'''
        label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        array = np.zeros((len(label_to_idx), len(labels)))
        rows = [label_to_idx[l] for l in labels]
        cols = range(len(labels))
        array[rows, cols] = 1

        return array

    @staticmethod
    def parse_filter(datalist):
        filtered = []
        for item in datalist:
            datum = Parses(item['sentence1_parse'], item['sentence2_parse'])
            filtered.append(datum)

        return filtered

    @staticmethod
    def binary_parse_filter(datalist):
        filtered = []
        for item in datalist:
            p1 = item['sentence1_binary_parse']
            p2 = item['sentence2_binary_parse']
            filtered.append(BinaryParses(p1, p2))

        return filtered

    @staticmethod
    def sentence_filter(datalist):
        filtered = []
        for item in datalist:
            filtered.append(Sentences(item['sentence1'], item['sentence2']))

        return filtered

    @staticmethod
    def text_filter(datalist):
        '''Uses datastream to extract and return all text in the stream'''
        acc = []
        for item in datalist:
            pair = item['sentence1'] + ' ' + item['sentence2']
            acc.append(pair)
        return ' '.join(acc)

    @staticmethod
    def xy_filter(datalist):
        filtered = []
        for item in datalist:
            s1 = item['sentence1']
            s2 = item['sentence2']
            filtered.append(TrainingPair(s1, s2, item['gold_label']))

        return filtered

    def load(self, filename):
        '''Returns a list of data items stored in named SNLI json file.'''
        data = []
        with open(self.path + filename) as f:
            for line in f:
                data.append(json.loads(line))

        return data

    def load_raw(self):
        self._train_data = self.load('snli_1.0_train.jsonl')
        self._dev_data = self.load('snli_1.0_dev.jsonl')
        self._test_data = self.load('snli_1.0_test.jsonl')

    def load_parses(self):
        '''Loads train/dev/test parse tree data for each item in SNLI'''
        self.load_raw()
        self._train_data = self.parse_filter(self._train_data)
        self._dev_data = self.parse_filter(self._dev_data)
        self._test_data = self.parse_filter(self._test_data)

    def load_binary_parses(self):
        '''Loads train/dev/test binary parse tree data for items in SNLI'''
        self.load_raw()
        self._train_data = self.binary_parse_filter(self._train_data)
        self._dev_data = self.binary_parse_filter(self._dev_data)
        self._test_data = self.binary_parse_filter(self._test_data)

    def load_sentences(self):
        '''Loads train/dev/test sentence data for each item in SNLI'''
        self.load_raw()
        self._train_data = self.sentence_filter(self._train_data)
        self._dev_data = self.sentence_filter(self._dev_data)
        self._test_data = self.sentence_filter(self._test_data)

    def load_xy_pairs(self):
        '''Loads train/dev/test sentences-label data for items in SNLI'''
        self.load_raw()
        self._train_data = self.xy_filter(self._train_data)
        self._dev_data = self.xy_filter(self._dev_data)
        self._test_data = self.xy_filter(self._test_data)

    def load_text(self):
        self.load_raw()
        self._train_data = self.text_filter(self._train_data)
        self._dev_data = self.text_filter(self._dev_data)
        self._test_data = self.text_filter(self._test_data)

    def build_vocab(self):
        '''Extract and build a vocab from all text in the corpus'''
        self.load_text()
        text = self.train_data + self.dev_data + self.test_data
        self.vocab = sorted(list(set(nltk.word_tokenize(text))))


class SICK(DataHandler):
    """An iterface to the SICK corpus for natural language inference. The
    dataset consists of 10,000 sentences pairs, and each pair is labelled with
    an inferential relationship and relatedness score. Because the dataset is
    small, streaming is not used; the data is all stored as a class attribute.

    Parameters:
    ----------
    path : str
        The absolute path to the directory containing the SICK corpus.

    Attributes:
    ----------
    entailment_pairs : list of namedtuples
        Each namedtuple has a field for sentence1, sentence2, entailment label
    relatedness_pairs : list of namedtuples
        Each namedtuple has a field for sentence1, sentence2, relation score
    data : list of dicts
        Each dict maps SICK fields to data for each sentence pair in corpus
    """
    def __init__(self, path):
        self.data = []

        with open(path) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                self.data.append(row)

        self._entailment_pairs = self.get_entailment_pairs()
        self._relatedness_pairs = self.get_relatedness_pairs()

    @property
    def entailment_pairs(self):
        return self._entailment_pairs

    @entailment_pairs.setter
    def entailment_pairs(self):
        raise Exception('SICK entailment data cannot be modified')

    @property
    def relatedness_pairs(self):
        return self._relatedness_pairs

    @relatedness_pairs.setter
    def relatedness_pairs(self):
        raise Exception('SICK relatedness data cannot be modified')

    def build_vocab(self):
        '''Extract and build a vocab from all text in the corpus'''
        text = self.get_text()
        self.vocab = sorted(list(set(nltk.word_tokenize(text))))

    def get_entailment_pairs(self):
        '''Return a list of sentence pairs with entailment labels'''
        pairs = []
        for item in self.data:
            s1 = item['sentence_A']
            s2 = item['sentence_B']
            label = item['entailment_judgment']
            pairs.append(TrainingPair(s1, s2, label))

        return pairs

    def get_relatedness_pairs(self):
        '''Return a list of sentence pairs with relatedness scores'''
        pairs = []
        for item in self.data:
            s1 = item['sentence_A']
            s2 = item['sentence_B']
            score = item['relatedness_score']
            pairs.append(RelationPair(s1, s2, score))

        return pairs

    def get_text(self):
        '''Returns a string containing all sentences in the SICK dataset'''
        acc = []
        for item in self.data:
            pair = item['sentence_A'] + ' ' + item['sentence_B']
            acc.append(pair.lower())

        return ' '.join(acc)
