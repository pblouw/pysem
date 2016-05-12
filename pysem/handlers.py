import os
import re
import json
import nltk
import platform
import cPickle as pickle
# import mputils
import filters

from sklearn.feature_extraction.text import CountVectorizer

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
        self.path = '/Users/peterblouw/corpora/snli_1.0/'
        self._reset_streams()

    def _stream(self, filename):
        '''A template generator for streaming from datasets'''
        with open(self.path + filename) as f:
            for line in f:
                yield json.loads(line)

    def _reset_streams(self, func=lambda x: x):
        '''Reset all generators that stream from datasets'''
        self.train_data = func(self._train_data())
        self.dev_data = func(self._dev_data())
        self.test_data = func(self._test_data())

    def _train_data(self):
        '''Generator that streams from training dataset'''
        return self._stream('snli_1.0_train.jsonl')

    def _dev_data(self):
        '''Generator that streams from development dataset'''
        return self._stream('snli_1.0_dev.jsonl')

    def _test_data(self):
        '''Generator that streams from testing dataset'''
        return self._stream('snli_1.0_test.jsonl')

    @property
    def extractor(self):
        return self._extraction_function

    @extractor.setter
    def extractor(self, func):
        self._reset_streams(func=func)
        self._extractor = func

    def build_vocab(self):
        train_text = self.all_text(self._train_data())
        dev_text = self.all_text(self._dev_data())
        test_text = self.all_text(self._test_data())
        all_text = train_text + dev_text + test_text

        with open('text.pickle', 'wb') as handle:
            pickle.dump(all_text, handle)

        tokens = nltk.word_tokenize(all_text)

        self.vocab = sorted(list(set(tokens)))

        with open('vocab.pickle', 'wb') as handle:
            pickle.dump(self.vocab, handle)

    def load_vocab(self):
        try:
            with open('vocab.pickle', 'rb') as handle:
                self.vocab = pickle.load(handle)
            with open('text.pickle', 'rb') as handle:
                self.text = pickle.load(handle)
        except:
            print 'No vocab file found!'

    @staticmethod
    def all_text(snli_stream):
        acc = []
        for snli_json in snli_stream:
            pair = snli_json['sentence1'] + ' ' + snli_json['sentence2']
            # pair = pair.encode('ascii', 'ignore').lower()
            acc.append(pair.lower())

        return ' '.join(acc)


if __name__ == '__main__':

    if platform.system() == 'Linux':
        corpuspath = '/home/pblouw/corpora/wikipedia'
        cachepath = '/home/pblouw/cache/'
    else:
        corpuspath = '/Users/peterblouw/corpora/wikipedia'
        cachepath = '/Users/peterblouw/cache/'

    snli = SNLI()
    snli.load_vocab()
    vectorizer = CountVectorizer(binary=True, vocabulary=set(snli.vocab))
    vectorizer.fit(snli.vocab)

    snli.extractor = filters.snli_sentences

    print len(vectorizer.get_feature_names())
    print len(snli.vocab)

    # sentences = filters.snli_sentences(snli.dev_data)
    # labels = filters.snli_labels(snli.dev_data)

    import numpy as np

    for _ in range(5):
        sample = next(snli.test_data)
        sent = sample[1].lower()
        print sent
        # print [w for w in sent if str(w) in vectorizer.get_feature_names()]
        x = vectorizer.transform([sent]).toarray()
        print x.shape
        print np.sum(x)
        print vectorizer.inverse_transform(x)
        # print np.where(x==1)
