import nltk
import re
import string
import collections
import multiprocessing as mp

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


def apply_async(func, paramlist):
    cpus = mp.cpu_count()
    pool = mp.Pool(processes=cpus)
    for _ in pool.map_async(func, paramlist).get():
        pass


def apply_return_async(func, paramlist):
    collect = []
    cpus = mp.cpu_count()
    pool = mp.Pool(processes=cpus)
    for _ in pool.map_async(func, paramlist).get():
        collect.append(_)
    pool.close()
    return collect


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
            cachefile.write(preprocessor(article) + ' DOCBREAK ')


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
