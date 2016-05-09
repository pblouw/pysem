import os
import re
import random
import nltk
import json

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class WikiHandler(object):
    """
    Builds a list of streams that stream sentences from random subsets of a
    Wikipedia dump (the location of which must be given to the constructor).
    """
    def __init__(self, corpuspath):
        self.corpuspath = corpuspath

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

    def cache(self, cachepath, streams):
        '''Caches preprocessed articles in a specified directory'''
        with open(cachepath+'cachtest.txt', 'w') as f:
            for stream in streams:
                for sen in stream:
                    f.write(sen + '\n')


class SnliHandler(object):
    '''Builds a list of streams of the Standford SNLI corpus'''
    def __init__(self, corpuspath):
        self.corpuspath = '/home/pblouw/corpora/snli_1.0'

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

    wikitext = SnliHandler(corpuspath)

    for example in wikitext.stream(100):
        print example['sentence1']
        print example['sentence2']
        print example['gold_label']
        print ''
