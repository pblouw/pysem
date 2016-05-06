import os 
import subprocess
import re
import random
import nltk
import string
import sys

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


class StreamGenerator(object):
    """
    Builds a list of streams that stream sentences from random subsets of a 
    Wikipedia dump (the location of which must be given to the constructor).
    """
    def __init__(self, corpuspath):
        self.corpuspath = corpuspath


    @staticmethod
    def preprocess(article):
        '''Perform basic preprocessing on Wikipedia article text'''
        text = re.sub("<.*>", "", article)
        text = text.decode('utf-8')
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


if __name__ == '__main__':

    corpuspath = '/Users/peterblouw/corpora/wikipedia'
    wikitext = StreamGenerator(corpuspath)
        
    streams = wikitext.build_streams(1, 1)

    for stream in streams:
        for sen in stream:
            print sen
