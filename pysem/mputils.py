import nltk
import re
import string
import collections

import multiprocessing as mp
import numpy as np

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


def apply_async(func, paramlist):
    acc = []
    cpus = mp.cpu_count()
    pool = mp.Pool(processes=cpus)
    for result in pool.map_async(func, paramlist).get():
        acc.append(result)
    pool.close()
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


def wiki_cache(params):
    '''Caches processed Wikipedia articles in a specified directory'''
    articles, cachepath, preprocessor = params
    with open(cachepath, 'w') as cachefile:
        for article in articles:
            cachefile.write(preprocessor(article) + '</doc>')


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
    counts = collections.Counter()
    sen_list = tokenizer.tokenize(article)
    sen_list = [s.replace('\n', ' ') for s in sen_list]
    sen_list = [s.translate(None, string.punctuation) for s in sen_list]
    sen_list = [s.translate(None, '1234567890') for s in sen_list]
    sen_list = [s.lower() for s in sen_list if len(s) > 5]
    sen_list = [nltk.word_tokenize(s) for s in sen_list]
    for sen in sen_list:
        counts.update(sen)

    return counts


def build_context_vectors(params):
    article, dimensions, base_vectors = params

    sen_list = tokenizer.tokenize(article)
    sen_list = [s.replace('\n', ' ') for s in sen_list]
    sen_list = [s.translate(None, string.punctuation) for s in sen_list]
    sen_list = [s.translate(None, '1234567890') for s in sen_list]
    sen_list = [s.lower() for s in sen_list if len(s) > 5]
    sen_list = [nltk.word_tokenize(s) for s in sen_list]
    sen_list = [[w for w in s if w in base_vectors.keys()] for s in sen_list]

    vectors = {key: np.zeros(dimensions) for key in base_vectors.keys()}
    for sen in sen_list:
        sen_sum = sum([base_vectors[word] for word in sen])
        for word in sen:
            word_sum = sen_sum - base_vectors[word]
            vectors[word] += word_sum

    return vectors
