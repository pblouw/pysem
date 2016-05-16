import nltk
import re
import multiprocessing as mp

tokenizer = nltk.load('tokenizers/punkt/english.pickle')


def parallelize(func, paramlist):
    cpus = mp.cpu_count()
    pool = mp.Pool(processes=cpus)
    for _ in pool.map_async(func, paramlist).get():
        pass


def wiki_cache(params):
    '''Caches processed Wikipedia articles in a specified directory'''
    articles, cachepath, preprocessor = params
    with open(cachepath, 'w') as cachefile:
        for article in articles:
            cachefile.write(preprocessor(article))


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
