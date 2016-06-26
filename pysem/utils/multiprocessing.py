import nltk
import re
import string
import collections

from pysem.utils.vsa import unitary_vector, HRR

import numpy as np
import multiprocessing as mp

tokenizer = nltk.load('tokenizers/punkt/english.pickle')

strip_pun = str.maketrans({key: None for key in string.punctuation})
strip_num = str.maketrans({key: None for key in '1234567890'})


def starmap(func, arglist):
    '''Apply a multiargument function to a list of input argument tuples'''
    acc = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in pool.starmap_async(func, arglist).get():
            acc.append(result)
    return acc


def plainmap(func, arglist):
    '''Apply a single argument function to a list of input arguments'''
    acc = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in pool.map_async(func, arglist).get():
            acc.append(result)
    return acc


def flatten(lst):
    '''Flattens arbitrarily nested lists'''
    acc = []
    for item in lst:
        if isinstance(item, list):
            acc += flatten(item)
        else:
            acc.append(item)
    return acc


def basic_strip(article):
    '''Strips out punctuation and very short sentences'''
    sen_list = tokenizer.tokenize(article)
    sen_list = [s.lower() for s in sen_list if len(s) > 5]
    return ' '.join(sen_list)


def max_strip(article):
    '''Strips out all sentences with brackets, numbers, etc.'''
    regex = re.compile(r"[0-9]|\[|\<|\(|\"|\'|\{")
    sen_list = tokenizer.tokenize(article)
    sen_list = [s for s in sen_list if not regex.findall(s)]
    sen_list = [s for s in sen_list if len(s) > 7]
    return ' '.join(sen_list)


def count_words(article):
    '''Builds a dictionary of word counts for an article'''
    counts = collections.Counter()
    sen_list = tokenizer.tokenize(article)
    sen_list = [s.replace('\n', ' ') for s in sen_list]
    sen_list = [s.translate(strip_pun) for s in sen_list]
    sen_list = [s.translate(strip_num) for s in sen_list]
    sen_list = [s.lower() for s in sen_list if len(s) > 5]
    sen_list = [nltk.word_tokenize(s) for s in sen_list]
    for sen in sen_list:
        counts.update(sen)

    return counts


class PoolData(object):

    def __init__(self, dim, vocab):
        self.vocab = vocab
        self.dim = dim
        self.vectors = np.random.normal(loc=0, scale=1/dim**0.5,
                                        size=(len(vocab), dim))

        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for idx, word in enumerate(vocab)}

        self.strip_pun = strip_pun
        self.strip_num = strip_num

        deps = set(['nsubj', 'dobj'])
        self.pos_i = [unitary_vector(self.dim) for i in range(5)]
        self.neg_i = [unitary_vector(self.dim) for i in range(5)]
        self.pos_i = dict((i, j) for i, j in enumerate(self.pos_i))
        self.neg_i = dict((i, j) for i, j in enumerate(self.neg_i))
        self.verb_deps = {dep: unitary_vector(self.dim) for dep in deps}

    def __getitem__(self, word):
        index = self.word_to_index[word]
        return HRR(self.vectors[index, :])

    def zeros(self):
        return np.zeros(self.dim)
