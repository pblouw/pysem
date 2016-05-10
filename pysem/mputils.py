import nltk

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.snowball.SnowballStemmer('english')


def wiki_cache(params):
    '''Caches processed Wikipedia articles in a specified directory'''
    articles, cachepath, filterfunc = params
    with open(cachepath, 'w') as cachefile:
        for article in articles:
            cachefile.write(filterfunc(article))


def basic_strip(article):
    '''Strips out punctuation and capitalization'''
    sen_list = tokenizer.tokenize(article)
    sen_list = [s.lower() for s in sen_list if len(s) > 5]
    return ' '.join(sen_list)
