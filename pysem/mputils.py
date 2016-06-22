import nltk
import re
import string
import collections

import multiprocessing as mp

tokenizer = nltk.load('tokenizers/punkt/english.pickle')

punc_translator = str.maketrans({key: None for key in string.punctuation})
num_translator = str.maketrans({key: None for key in '1234567890'})


def starmap(func, paramlist):
    acc = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in pool.starmap_async(func, paramlist).get():
            acc.append(result)
    return acc


def plainmap(func, paramlist):
    acc = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in pool.map_async(func, paramlist).get():
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
    '''Strips out punctuation and capitalization'''
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
    sen_list = [s.translate(punc_translator) for s in sen_list]
    sen_list = [s.translate(num_translator) for s in sen_list]
    sen_list = [s.lower() for s in sen_list if len(s) > 5]
    sen_list = [nltk.word_tokenize(s) for s in sen_list]
    for sen in sen_list:
        counts.update(sen)

    return counts
