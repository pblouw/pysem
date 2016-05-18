import os
import re
import json
import nltk
import cPickle as pickle
import numpy as np

from collections import Counter
from mputils import apply_async, wiki_cache, count_words

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class DataHandler(object):
    """Base class for handling datasets"""
    def __init__(self, path):
        self.path = path
        self._reset_streams()

    def _reset_streams(self):
        raise Exception('DataHandlers must implement a reset method')

    def save_vocab(self, filename):
        with open(filename + '.pickle', 'wb') as pfile:
            pickle.dump(self.vocab, pfile)

    def load_vocab(self, filename):
        '''Load a vocab that has been previously built'''
        try:
            with open(filename + '.pickle', 'rb') as pfile:
                self.vocab = pickle.load(pfile)
        except:
            print 'No vocab file with that name was found!'

    @property
    def extractor(self):
        return self._extractor

    @extractor.setter
    def extractor(self, func):
        self._reset_streams(func=func)
        self._extractor = func


class Wikipedia(DataHandler):
    """
    Provides various options for processing and handling text from a
    Wikipedia dump (which must minimally preprocessed to remove HTML).
    """
    def __init__(self, path, article_limit=None, from_cache=False):
        self.path = path
        self.article_limit = article_limit if article_limit else np.inf
        self.from_cache = from_cache
        self._reset_streams()
        self._article_count = 0

    def _reset_streams(self):
        self.batches = self._batches()
        self.articles = self._articles()
        self.sentences = self._sentences()

    def _batches(self):
        for root, dirs, files in os.walk(self.path):
            for fname in files:
                with open(root + '/' + fname, 'r') as f:
                    articles = f.read().split('</doc>')
                    if not self.from_cache:
                        articles = [self.preprocess(a) for a in articles]

                    yield [a for a in articles if len(a) > 0]

    def _articles(self):
        for articles in self._batches():
            for article in articles:
                if self._article_count >= self.article_limit:
                    raise StopIteration()
                yield article

                self._article_count += 1

    def _sentences(self):
        for article in self._articles():
            sents = tokenizer.tokenize(article)
            sents = [s.replace('\n', '') for s in sents]
            for s in sents:
                yield s

    def write_to_cache(self, path, parsefunc, batchsize=10):
        '''Write batches of preprocessed articles to cache for later use'''
        paramlist = []
        articles = []
        count = 0

        for article in self.articles:
            articles.append(article)

            if len(articles) == 200:
                fname = str(count) + '.txt'
                params = (articles, path + fname, parsefunc)
                paramlist.append(params)
                articles = []
                count += 1

            if len(paramlist) == batchsize:
                apply_async(wiki_cache, paramlist)
                paramlist = []

        if len(paramlist) != 0:
            apply_async(wiki_cache, paramlist)

        self._reset_streams()

    def build_vocab(self, cutoff=0.5, batchsize=100):
        counter = Counter()
        articles = []
        for article in self.articles:
            articles.append(article)
            print len(articles)
            if len(articles) == batchsize:
                for counts in apply_async(count_words, articles):
                    counter.update(counts)
                articles = []

        self.vocab = counter.most_common(int(len(counter)*cutoff))
        self.vocab = sorted([pair[0] for pair in self.vocab])
        self._reset_streams()

    @staticmethod
    def preprocess(document):
        '''Perform basic preprocessing on a Wikipedia document'''
        document = re.sub("<.*>|<|>", "", document)

        try:
            document = document.decode('unicode_escape')
            document = document.encode('ascii', 'ignore')
        except UnicodeDecodeError:
            print 'UnicodeDecodeError'
            return str()

        document = document.split('\n')[3:]
        document = ' '.join(document)
        return document


class SNLI(DataHandler):
    """Extracts data from the SNLI corpus"""
    def _reset_streams(self, func=lambda x: x):
        '''Reset all generators that stream from datasets'''
        self.train_data = func(self._train_data())
        self.dev_data = func(self._dev_data())
        self.test_data = func(self._test_data())

    def _stream(self, filename):
        with open(self.path + filename) as f:
            for line in f:
                yield json.loads(line)

    def _train_data(self):
        return self._stream('snli_1.0_train.jsonl')

    def _dev_data(self):
        return self._stream('snli_1.0_dev.jsonl')

    def _test_data(self):
        return self._stream('snli_1.0_test.jsonl')

    def build_vocab(self):
        '''Extract and build a vocab from all text in the corpus'''
        self.extractor = self.get_text
        text = self.train_data + self.dev_data + self.test_data
        text = text.encode('ascii')
        self.vocab = set(nltk.word_tokenize(text))
        self._reset_streams()

    @staticmethod
    def get_text(stream):
        '''Modifies datastream to extract all text in the stream'''
        acc = []
        for item in stream:
            pair = item['sentence1'] + ' ' + item['sentence2']
            acc.append(pair.lower())
        return ' '.join(acc)

    @staticmethod
    def get_sentences(stream):
        '''Modifies datastream to yield sentence pairs'''
        for item in stream:
            yield (item['sentence1'], item['sentence2'])

    @staticmethod
    def get_xy_pairs(stream):
        '''Modifies datastream to yield x,y pairs for model training'''
        for item in stream:
            x = (item['sentence1'], item['sentence2'])
            y = item['gold_label']
            yield (x, y)

    @staticmethod
    def get_parses(stream):
        '''Modifies datastream to yield parses of sentence pairs'''
        for item in stream:
            p1 = item['sentence1_parse']
            p2 = item['sentence2_parse']
            yield (p1, p2)

    @staticmethod
    def get_binary_parses(stream):
        '''Modifies datastream to yield binary parsies of sentence pairs'''
        for item in stream:
            p1 = item['sentence1_binary_parse']
            p2 = item['sentence2_binary_parse']
            yield (p1, p2)
