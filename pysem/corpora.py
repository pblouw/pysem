import os
import re
import json
import sys
import nltk
import pickle
import itertools

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
                    for a in articles:
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
        self.reset_streams()

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

    @property
    def extractor(self):
        return self._extractor

    @extractor.setter
    def extractor(self, func):
        self.reset_streams(func=func)
        self._extractor = func

    @staticmethod
    def get_parses(stream):
        '''Modifies datastream to yield parses of sentence pairs'''
        for item in stream:
            yield Parses(item['sentence1_parse'], item['sentence2_parse'])

    @staticmethod
    def get_binary_parses(stream):
        '''Modifies datastream to yield binary parses of sentence pairs'''
        for item in stream:
            p1 = item['sentence1_binary_parse']
            p2 = item['sentence2_binary_parse']
            yield BinaryParses(p1, p2)

    @staticmethod
    def get_text(stream):
        '''Uses datastream to extract and return all text in the stream'''
        acc = []
        for item in stream:
            pair = item['sentence1'] + ' ' + item['sentence2']
            acc.append(pair.lower())
        return ' '.join(acc)

    @staticmethod
    def get_sentences(stream):
        '''Modifies datastream to yield sentence pairs'''
        for item in stream:
            yield Sentences(item['sentence1'], item['sentence2'])

    @staticmethod
    def get_xy_pairs(stream):
        '''Modifies datastream to yield x,y pairs for model training'''
        for item in stream:
            s1 = item['sentence1']
            s2 = item['sentence2']
            yield TrainingPair(s1, s2, item['gold_label'])

    def build_vocab(self):
        '''Extract and build a vocab from all text in the corpus'''
        self.extractor = self.get_text
        text = self.train_data + self.dev_data + self.test_data
        self.vocab = sorted(list(set(nltk.word_tokenize(text))))
        self.reset_streams()

    def reset_streams(self, func=lambda x: x):
        '''Reset all generators that stream data from the SNLI dump'''
        self._train_data = func(self._train_stream())
        self._dev_data = func(self._dev_stream())
        self._test_data = func(self._test_stream())

    def _stream(self, filename):
        with open(self.path + filename) as f:
            for line in f:
                yield json.loads(line)

    def _train_stream(self):
        return self._stream('snli_1.0_train.jsonl')

    def _dev_stream(self):
        return self._stream('snli_1.0_dev.jsonl')

    def _test_stream(self):
        return self._stream('snli_1.0_test.jsonl')
