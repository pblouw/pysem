import os
import re
import json
import nltk
import cPickle as pickle

from collections import Counter
from mputils import apply_async, wiki_cache, count_words

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class DataHandler(object):
    """Base class for handling datasets"""
    def __init__(self, path):
        self._path = path
        self._reset_streams()

    def _reset_stream(self):
        raise Exception('DataHandlers must immplement a reset method')

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
    def _reset_streams(self):
        self.batches = self._batches()
        self.articles = self._articles()
        self.sentences = self._sentences()

    def _batches(self):
        for root, dirs, files in os.walk(self._path):
            for file in files:
                with open(root + '/' + file, 'r') as f:
                    articles = f.read().split('</doc>')
                    articles = [self.preprocess(a) for a in articles]
                    yield [a for a in articles if len(a) > 0]

    def _articles(self):
        for articles in self._batches():
            for article in articles:
                yield article

    def _sentences(self):
        for article in self._articles():
            sents = tokenizer.tokenize(article)
            sents = [s.replace('\n', '') for s in sents]
            for s in sents:
                yield s

    def _cache_batches(self):
        for root, dirs, files in os.walk(self._path):
            for file in files:
                with open(root + '/' + file, 'r') as f:
                    articles = f.read().split(' </doc> ')
                    yield [a for a in articles if len(a) > 0]

    def _cache_articles(self):
        for articles in self._cache_batches():
            for article in articles:
                yield article

    def _cache_sentences(self):
        for article in self._cache_articles():
            sents = tokenizer.tokenize(article)
            sents = [s.replace('\n', '') for s in sents]
            for s in sents:
                yield s

    def write_to_cache(self, path, parsefunc, n_batches, batchsize=100):
        '''Write batches of preprocessed articles to cache for later use'''
        for _ in xrange(n_batches):
            paramlist = []
            for __ in xrange(batchsize):
                fname = str(_) + str(__) + '.txt'
                paramlist.append((next(self.batches), path+fname, parsefunc))

            result = apply_async(wiki_cache, paramlist)

    def load_from_cache(self, path):
        self._path = path
        self.batches = self._cache_batches()
        self.articles = self._cache_articles()
        self.sentences = self._cache_sentences()

    def build_vocab(self, cutoff=0.5):
        counter = Counter()
        for articles in self.batches:
            for counts in apply_async(count_words, articles):
                counter.update(counts)

        self.vocab = counter.most_common(int(len(counter)*cutoff))
        return self.vocab

    @staticmethod
    def preprocess(document):
        '''Perform basic preprocessing on a Wikipedia document'''
        document = re.sub("<.*>|<|>", "", document)

        try:
            document = document.decode('unicode-escape')
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
        with open(self._path + filename) as f:
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

        with open('vocab.pickle', 'wb') as pfile:
            pickle.dump(self.vocab, pfile)

    def load_vocab(self):
        '''Load a vocab that has been previously built'''
        try:
            with open('vocab.pickle', 'rb') as pfile:
                self.vocab = pickle.load(pfile)
        except:
            print 'No vocab file found!'

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
