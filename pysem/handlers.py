import os
import re
import json
import nltk
import platform
# import mputils
import filters

# import multiprocessing as mp

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class DataHandler(object):
    """Base class for handling datasets"""
    def __init__(self):
        pass


class Wikipedia(DataHandler):
    """
    Provides various options for processing and handling text from a
    Wikipedia dump (which must minimally preprocessed to remove HTML).
    """
    def __init__(self, path):
        self._path = path
        self.initialize_streams()

    def initialize_streams(self):
        '''Reset all generators that stream docs/sents/batches'''
        self.batches = self._batches()
        self.documents = self._documents()
        self.sentences = self._sentences()

    def _batches(self):
        '''Generator that streams batches of documents from Wikipedia'''
        for root, dirs, files in os.walk(self._path):
            for file in files:
                with open(root + '/' + file, 'r') as f:
                    documents = f.read().split('</doc>')
                    documents = [self.preprocess(d) for d in documents]
                    yield [d for d in documents if len(d) > 0]

    def _documents(self):
        '''Generator that streams single documents from Wikipedia'''
        for docbatch in self._batches():
            for doc in docbatch:
                yield doc

    def _sentences(self):
        '''Generator that streams sentences from Wikipedia'''
        for doc in self._documents():
            sents = tokenizer.tokenize(doc)
            sents = [s.replace('\n', '') for s in sents]
            for s in sents:
                yield s

    @property
    def batch_filter(self):
        return self._batch_filter

    @property
    def document_filter(self):
        return self._document_filter

    @property
    def sentence_filter(self):
        return self._sentence_filter

    @batch_filter.setter
    def batch_filter(self, filterfunc):
        self.batches = filterfunc(self.batches)
        self._batch_filter = filterfunc

    @document_filter.setter
    def document_filter(self, filterfunc):
        self.documents = filterfunc(self.documents)
        self._document_filter = filterfunc

    @sentence_filter.setter
    def sentence_filter(self, filterfunc):
        self.sentences = filterfunc(self.sentences)
        self._sentence_filter = filterfunc

    @staticmethod
    def preprocess(document):
        '''Perform basic preprocessing on a Wikipedia document'''
        document = re.sub("<.*>|<|>", "", document)
        document = document.decode('utf-8')
        document = document.encode('ascii', 'ignore')
        document = document.split('\n')[3:]
        document = ' '.join(document)
        return document


class SNLI(DataHandler):
    '''Extracts data from the SNLI corpus'''
    def __init__(self):
        self.path = '/home/pblouw/corpora/snli_1.0/'
        self.initialize_streams()

    def initialize_streams(self):
        '''Reset all generators that stream corpus data'''
        self.train_set = self._train_set()
        self.dev_set = self._dev_set()
        self.test_set = self._test_set()

    def _train_set(self):
        '''Generator that streams training data from the SNLI corpus'''
        with open(self.path + 'snli_1.0_train.jsonl') as f:
            for line in f:
                yield json.loads(line)

    def _dev_set(self):
        '''Generator that streams dev data from the SNLI corpus'''
        with open(self.path + 'snli_1.0_dev.jsonl') as f:
            for line in f:
                yield json.loads(line)

    def _test_set(self):
        '''Generator that streams test data from the SNLI corpus'''
        with open(self.path + 'snli_1.0_test.jsonl') as f:
            for line in f:
                yield json.loads(line)

    @property
    def train_filter(self):
        return self._train_filter

    @property
    def dev_filter(self):
        return self._dev_filter

    @property
    def test_filter(self):
        return self._test_filter

    @property
    def batchsize(self):
        return self._batchsize

    @train_filter.setter
    def train_filter(self, filterfunc):
        self.train_set = filterfunc(self.train_set)
        self._train_filter = filterfunc

    @dev_filter.setter
    def dev_filter(self, filterfunc):
        self.dev_set = filterfunc(self.dev_set)
        self._dev_filter = filterfunc

    @test_filter.setter
    def test_filter(self, filterfunc):
        self.test_set = filterfunc(self.test_set)
        self._test_filter = filterfunc

    @batchsize.setter
    def batchsize(self, n):
        self._batchsize = n
        self.train_set = self.collect(self.train_set, n)
        self.dev_set = self.collect(self.dev_set, n)
        self.test_set = self.collect(self.test_set, n)

    @staticmethod
    def collect(stream, batchsize):
        while True:
            acc = []
            for _ in xrange(batchsize):
                acc.append(next(stream))
            yield acc


if __name__ == '__main__':

    if platform.system() == 'Linux':
        corpuspath = '/home/pblouw/corpora/wikipedia'
        cachepath = '/home/pblouw/cache/'
    else:
        corpuspath = '/Users/peterblouw/corpora/wikipedia'
        cachepath = '/Users/peterblouw/cache/'

    snli = SNLI()
    # snli.test_filter = mputils.sentence_filter.

    sentences = filters.snli_sentences(snli._dev_set())
    labels = filters.snli_labels(snli._dev_set())

    for _ in range(100):
        print sentences.next()
        print labels.next()
        print ''
