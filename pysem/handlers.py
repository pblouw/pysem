import os
import re
import nltk
import platform
import mputils

import multiprocessing as mp

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class CorpusHandler(object):
    """Base class for handling linguistic corpora"""
    def __init__(self, corpuspath):
        pass


class Wikipedia(CorpusHandler):
    """
    Provides various options for processing and handling text from a
    Wikipedia dump (which must minimally preprocessed to remove HTML).
    """
    def __init__(self, corpuspath):
        self._corpuspath = corpuspath
        self._batch_filter = self.default_filter
        self._document_filter = self.default_filter
        self._sentence_filter = self.default_filter
        self.initialize_streams()

    @staticmethod
    def default_filter(stream):
        return stream

    @staticmethod
    def preprocess(document):
        '''Perform basic preprocessing on a Wikipedia document'''
        document = re.sub("<.*>|<|>", "", document)
        document = document.decode('utf-8')
        document = document.encode('ascii', 'ignore')
        document = document.split('\n')[3:]
        document = ' '.join(document)
        return document

    def _batches(self):
        '''Generator that streams batches of documents from Wikipedia'''
        for root, dirs, files in os.walk(self._corpuspath):
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

    def initialize_streams(self):
        '''Reset all generators that stream docs/sents/batches'''
        self.batches = self._batches()
        self.documents = self._documents()
        self.sentences = self._sentences()


# class SNLI(CorpusHandler):
#     '''Builds a list of streams of the Standford SNLI corpus'''
#     def __init__(self, corpuspath):
#         self.corpuspath = '/Users/peterblouw/corpora/snli_1.0'

#     def stream(self, n_examples):
#         '''Generator that streams data from SNLI corpus'''
#         with open(self.corpuspath + '/snli_1.0_train.jsonl') as f:
#             for line in f:
#                 yield json.loads(line)


if __name__ == '__main__':

    if platform.system() == 'Linux':
        corpuspath = '/home/pblouw/corpora/wikipedia'
        cachepath = '/home/pblouw/cache/'
    else:
        corpuspath = '/Users/peterblouw/corpora/wikipedia'
        cachepath = '/Users/peterblouw/cachec/'

    wp = Wikipedia(corpuspath)

    params = []
    for _ in range(100):
        params.append((next(wp.batches),
                       cachepath+str(_),
                       mputils.basic_strip))

    output = mp.Pool().map_async(mputils.wiki_cache, params)
    for _ in output.get():
        continue
