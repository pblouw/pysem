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
    '''Apply a multiargument function to a list of input argument tuples.'''
    acc = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in pool.starmap_async(func, arglist).get():
            acc.append(result)
    return acc


def plainmap(func, arglist):
    '''Apply a single argument function to a list of input arguments.'''
    acc = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in pool.map_async(func, arglist).get():
            acc.append(result)
    return acc


def flatten(lst):
    '''Flattens arbitrarily nested lists.'''
    acc = []
    for item in lst:
        if isinstance(item, list):
            acc += flatten(item)
        else:
            acc.append(item)
    return acc


def basic_strip(article):
    '''Strips out punctuation and very short sentences.'''
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
    '''Builds a dictionary of word counts for an article.'''
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


class PoolContainer(object):
    """ A container for readonly data that is shared across processes in a
    multiprocessing pool. Use of this container object eliminates the need to
    create a copy of the shared data for each process, which in turn allows
    for the use of multiprocessing in cases that would otherwise require
    a prohibitive amount of memory. PoolContainer objects must be defined as
    global variables to be used with multiprocessing pools.

    Parameters:
    ----------
    dim : int
        A constant dimensionality variable to be shared across processes.
    vocab : list
        A constant vocabulary of words to be shared across processes.
    winsize : int
        A constant windowsize to be shared across processes computing
        order embeddings in parallel. Defaults to 5.
    """
    def __init__(self, dim, vocab, winsize=2):
        self.dim = dim
        self.vocab = vocab
        self.winsize = winsize
        self.vectors = np.random.normal(0, 1/dim**0.5, (len(vocab), dim))

        self.build_text_strippers()
        self.build_word_maps()

    def __getitem__(self, word):
        idx = self.word_to_idx[word]
        return HRR(self.vectors[idx, :])

    def build_dependency_tags(self):
        deps = set(['nsubj', 'dobj'])
        self.verb_deps = {dep: unitary_vector(self.dim) for dep in deps}

    def build_position_tags(self):
        pos_idx = [unitary_vector(self.dim) for i in range(self.winsize)]
        neg_idx = [unitary_vector(self.dim) for i in range(self.winsize)]
        self.pos_idx = dict((i, j) for i, j in enumerate(pos_idx))
        self.neg_idx = dict((i, j) for i, j in enumerate(neg_idx))

    def build_text_strippers(self):
        self.strip_pun = strip_pun
        self.strip_num = strip_num

    def build_word_maps(self):
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def zeros(self):
        '''Returns an array of zeros for use in initializing defaultdicts'''
        return np.zeros(self.dim)
