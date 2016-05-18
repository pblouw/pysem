import nltk
import string
import operator
import trainglobals

import numpy as np
import multiprocessing as mp

from nltk.stem.snowball import SnowballStemmer

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')


class EmbeddingModel(object):
    """Base class for embedding models"""
    def __init__(self):
        pass

    @staticmethod
    def normalize(v):
        if np.linalg.norm(v) > 0:
            return v / np.linalg.norm(v)

punc_translator = str.maketrans({key: None for key in string.punctuation})
num_translator = str.maketrans({key: None for key in '1234567890'})


def build_context_vectors(article):
    # article, dimensions, base_vectors = params

    sen_list = tokenizer.tokenize(article)
    sen_list = [s.replace('\n', ' ') for s in sen_list]
    sen_list = [s.translate(punc_translator) for s in sen_list]
    sen_list = [s.translate(num_translator) for s in sen_list]
    sen_list = [s.lower() for s in sen_list if len(s) > 5]
    sen_list = [nltk.word_tokenize(s) for s in sen_list]
    sen_list = [[w.lower() for w in s]
                for s in sen_list if len(s) > 4]
    sen_list = [[w for w in s if w in trainglobals.vocab] for s in sen_list]
    vec_dict = dict()

    for sen in sen_list:
        sen_sum = sum([trainglobals.v_vecs[trainglobals.vocab_dict[w], :]
                       for w in sen])
        for word in sen:
            w_index = trainglobals.vocab_dict[word]
            w_sum = sen_sum - trainglobals.v_vecs[w_index, :]
            if np.linalg.norm(w_sum) == 0.0:
                continue
            else:
                try:
                    vec_dict[word] += w_sum
                except KeyError:
                    vec_dict[word] = w_sum
    return vec_dict


class RandomIndexing(EmbeddingModel):

    def __init__(self, corpus):
        self._corpus = corpus

    def train(self, dimensions, voc, preprocessing=None):
        self.dim = dimensions
        vocab = voc

        def get_synonyms(word):
            probe = w_vecs[trainglobals.vocab_dict[word], :]
            rank_words(np.dot(w_vecs, probe))

        def rank_words(comparison):
            rank = zip(range(len(vocab)), comparison)
            rank = sorted(rank, key=operator.itemgetter(1), reverse=True)
            top_words = [(word_dict[item[0]], item[1]) for item in rank[:10]]

            for word in top_words[:5]:
                print(word[0], word[1])

        trainglobals.vocab = {w for w in vocab if w not in stopwords}
        trainglobals.vocab_dict = {word: ind for ind, word in
                                   enumerate(trainglobals.vocab)}
        word_dict = {ind: word for word, ind
                     in trainglobals.vocab_dict.items()}

        w_vecs = np.zeros((len(vocab), self.dim))
        trainglobals.v_vecs = np.random.normal(loc=0, scale=1/(self.dim**0.5),
                                               size=(len(vocab), self.dim))

        articles = []
        for article in self._corpus.articles:
            articles.append(article)

            if len(articles) == 200:
                cpus = mp.cpu_count()
                pool = mp.Pool(processes=cpus)
                result = pool.map_async(build_context_vectors, articles)
                for r in result.get():
                    for i, j in r.items():
                        w_vecs[trainglobals.vocab_dict[i], :] += j

                articles = []

        for word in trainglobals.vocab:
            w_index = trainglobals.vocab_dict[word]
            if np.all(w_vecs[w_index, :] == 0):
                w_vecs[w_index, :] = trainglobals.v_vecs[w_index, :]

        norms = np.linalg.norm(w_vecs, axis=1)
        w_vecs = np.divide(w_vecs, norms[:, np.newaxis])

        word_list = ['song', 'went', 'drive', 'play', 'beer',
                     'gun', 'red', 'king', 'queen']
        query_list = word_list

        for query in query_list:
            print('Nearest neighbors to "%s":' % query)
            get_synonyms(query)
            print('')

        # return context_vectors

    def unit_vector(self):
        return self.normalize(np.random.normal(loc=0, size=1))


class SkipGram(EmbeddingModel):
    pass


class CBOW(EmbeddingModel):
    pass


class Word2Vec(SkipGram, CBOW):
    pass
