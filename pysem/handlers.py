import os
import re
import json
import nltk
import cPickle as pickle

from mputils import parallelize, wiki_cache

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

    def write_to_cache(self, path, preprocessor, n_batches):
        paramlist = []
        for _ in xrange(n_batches):
            paramlist.append((next(self.batches),
                              path + str(_) + '.txt',
                              preprocessor))

        parallelize(wiki_cache, paramlist)

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
    """Extracts data from the SNLI corpus"""
    def _reset_streams(self, func=lambda x: x):
        '''Reset all generators that stream from datasets'''
        self.train_data = func(self._train_data())
        self.dev_data = func(self._dev_data())
        self.test_data = func(self._test_data())

    def _stream(self, filename):
        '''A template generator for streaming from datasets'''
        with open(self._path + filename) as f:
            for line in f:
                yield json.loads(line)

    def _train_data(self):
        '''Generator that streams from training dataset'''
        return self._stream('snli_1.0_train.jsonl')

    def _dev_data(self):
        '''Generator that streams from development dataset'''
        return self._stream('snli_1.0_dev.jsonl')

    def _test_data(self):
        '''Generator that streams from testing dataset'''
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
        acc = []
        for item in stream:
            pair = item['sentence1'] + ' ' + item['sentence2']
            acc.append(pair.lower())
        return ' '.join(acc)

    @staticmethod
    def get_sentences(stream):
        for item in stream:
            yield (item['sentence1'], item['sentence2'])

    @staticmethod
    def get_xy_pairs(stream):
        for item in stream:
            x = (item['sentence1'], item['sentence2'])
            y = item['gold_label']
            yield (x, y)

    @staticmethod
    def get_parses(stream):
        for item in stream:
            p1 = item['sentence1_parse']
            p2 = item['sentence2_parse']
            yield (p1, p2)

    @staticmethod
    def get_binary_parses(stream):
        for item in stream:
            p1 = item['sentence1_binary_parse']
            p2 = item['sentence2_binary_parse']
            yield (p1, p2)
