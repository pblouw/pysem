import os
import re
import json
import nltk
import pickle
import itertools

import numpy as np

from collections import Counter
from pysem.mputils import apply_async, starmap, count_words

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
            print('No vocab file with that name was found!')

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

    @staticmethod
    def preprocess(article):
        '''Perform basic preprocessing on a Wikipedia article'''
        article = re.sub("<.*>|<|>", "", article)
        article = article.split('\n')[3:]  # removes title from 1st sentence
        article = ' '.join(article)
        return article

    @staticmethod
    def cache(articles, process, path):
        '''Caches modified Wikipedia articles in a specified directory'''
        with open(path, 'w') as cachefile:
            for article in articles:
                cachefile.write(process(article) + '</doc>')

    def _reset_streams(self):
        self.article_count = 0
        self.batches = self._batches()
        self.articles = self._articles()
        self.sentences = self._sentences()

    def _batches(self):
        for root, dirs, files in os.walk(self.path):
            for fname in files:
                fpath = root + '/' + fname
                with open(fpath, 'r', encoding='ascii', errors='ignore') as f:
                    articles = f.read().split('</doc>')

                    if not self.from_cache:
                        articles = [self.preprocess(a) for a in articles]

                    yield [a for a in articles if len(a) > 0]

    def _articles(self):
        for articles in self._batches():
            for article in articles:
                yield article

                self.article_count += 1
                if self.article_count >= self.article_limit:
                    raise StopIteration()

    def _sentences(self):
        for article in self._articles():
            sents = tokenizer.tokenize(article)
            sents = [s.replace('\n', '') for s in sents]
            for s in sents:
                yield s

    def write_to_cache(self, path, process, n_per_file=200, pool_size=10):
        '''Write batches of processed articles to cache for later use'''
        paramlist = []
        for count in itertools.count(0):
            batch = list(itertools.islice(self.articles, n_per_file))
            fname = str(count) + '.txt'
            paramlist.append((batch, process, path + fname))

            if count % pool_size == 0 and count != 0:
                starmap(self.cache, paramlist)
                paramlist = []

            elif len(batch) < n_per_file:
                starmap(self.cache, paramlist)
                break

        self._reset_streams()

    def build_vocab(self, cutoff=0.5, batchsize=100):
        counter = Counter()
        articles = []
        for article in self.articles:
            articles.append(article)
            if len(articles) == batchsize:
                for counts in apply_async(count_words, articles):
                    counter.update(counts)
                articles = []

        self.vocab = counter.most_common(int(len(counter)*cutoff))
        self.vocab = sorted([pair[0] for pair in self.vocab])
        self._reset_streams()


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
