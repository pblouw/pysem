import os
import re
import random
import nltk
import json

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class CorpusHandler(object):
    """Base class for handling linguistic corpora"""
    def __init__(self, corpuspath):
        self.corpuspath = corpuspath

    def cache(self, cachepath, stream, maxsize=10, count=0):
        '''Caches stream output in a specified directory'''
        abspath = cachepath + 'cache_' + str(count) + '.txt'

        with open(abspath, 'w') as cachefile:
            while True:
                try:
                    cachefile.write(next(stream))
                except StopIteration:
                    return
                if os.path.getsize(abspath) > maxsize * 1000000:
                    break

        self.cache(cachepath, stream, count=count+1)


class Wikipedia(CorpusHandler):
    """
    Builds a list of streams that stream sentences from random subsets of a
    Wikipedia dump (the location of which must be given to the constructor).
    """
    @staticmethod
    def preprocess(article):
        '''Perform basic preprocessing on Wikipedia article text'''
        text = re.sub("<.*>|<|>", "", article)
        text = text.decode('unicode_escape')
        text = text.encode('ascii', 'ignore')
        text = text.split('\n')[3:]
        text = ' '.join(text)
        sen_list = tokenizer.tokenize(text)
        sen_list = [s.replace('\n', ' ') for s in sen_list]

        return sen_list

    def stream(self, files):
        '''Generator that streams sentences from Wikipedia'''
        for file in files:
            with open(file, 'r') as f:
                articles = f.read().split('</doc>')
                for article in articles:
                    sen_list = self.preprocess(article)
                    for sen in sen_list:
                        yield sen

    def build_streams(self, n_streams, n_files, randomize=True):
        '''Build a set of generators for streaming Wikipedia text'''
        file_paths = set()
        for root, dirs, files in os.walk(self.corpuspath):
            for file in files:
                file_paths.add(root+'/'+file)

        streams = list()
        for _ in xrange(n_streams):
            path_set = random.sample(file_paths, n_files)
            if not randomize:
                file_paths = file_paths.difference_update(path_set)
            streams.append(self.stream(path_set))

        return streams


class SNLI(CorpusHandler):
    '''Builds a list of streams of the Standford SNLI corpus'''
    def __init__(self, corpuspath):
        self.corpuspath = '/Users/peterblouw/corpora/snli_1.0'

    def stream(self, n_examples):
        '''Generator that streams data from SNLI corpus'''
        with open(self.corpuspath + '/snli_1.0_train.jsonl') as f:
            for line in f:
                yield json.loads(line)
                n_examples -= 1
                if n_examples == 0:
                    break


if __name__ == '__main__':

    corpuspath = '/Users/peterblouw/corpora/wikipedia'
    cachepath = '/Users/peterblouw/'

    wikitext = Wikipedia(corpuspath)

    stream = wikitext.build_streams(1, 10).pop()
    wikitext.cache(cachepath+'cache/', stream)
