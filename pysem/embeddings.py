import nltk
import string
import operator

import numpy as np
import multiprocessing as mp

from nltk.stem.snowball import SnowballStemmer
from .mputils import Container

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')

glb = Container()

punc_translator = str.maketrans({key: None for key in string.punctuation})
num_translator = str.maketrans({key: None for key in string.digits})


class EmbeddingModel(object):
    """Base class for embedding models"""
    def __init__(self):
        pass

    @staticmethod
    def normalize(v):
        if np.linalg.norm(v) > 0:
            return v / np.linalg.norm(v)


class RandomIndexing(EmbeddingModel):

    def __init__(self, corpus):
        self._corpus = corpus

    @staticmethod
    def build_order_vectors(article):
        sen_list = tokenizer.tokenize(article)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(punc_translator) for s in sen_list]
        sen_list = [s.translate(num_translator) for s in sen_list]
        sen_list = [s.lower() for s in sen_list if len(s) > 5]
        sen_list = [nltk.word_tokenize(s) for s in sen_list]
        sen_list = [[w.lower() for w in s] for s in sen_list]
        sen_list = [[w for w in s if w in glb.vocab] for s in sen_list]
        vec_dict = dict()
        maxn = 5
        for sen in sen_list:
            pos = 0
            for x in range(len(sen)):
                o_sum = np.zeros(glb.dim)
                for y in range(maxn):
                    if pos+y+1 < len(sen):
                        w = glb.base_vectors[glb.word_to_idx[sen[pos+y+1]], :]
                        p = glb.pos_i[y]
                        o_sum += glb.convolve(w, p)
                    if pos-y-1 >= 0:
                        w = glb.base_vectors[glb.word_to_idx[sen[pos-y-1]], :]
                        p = glb.neg_i[y]
                        o_sum += glb.convolve(w, p)

                word = sen[x]
                try:
                    vec_dict[word] += o_sum
                except KeyError:
                    vec_dict[word] = o_sum
                pos += 1

        return vec_dict

    @staticmethod
    def build_context_vectors(article):
        sen_list = tokenizer.tokenize(article)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(punc_translator) for s in sen_list]
        sen_list = [s.translate(num_translator) for s in sen_list]
        sen_list = [s.lower() for s in sen_list if len(s) > 5]
        sen_list = [nltk.word_tokenize(s) for s in sen_list]
        sen_list = [[w.lower() for w in s] for s in sen_list]
        sen_list = [[w for w in s if w in glb.vocab] for s in sen_list]
        vec_dict = dict()

        for sen in sen_list:
            sen_sum = sum([glb.base_vectors[glb.word_to_idx[w], :]
                           for w in sen])
            for word in sen:
                w_index = glb.word_to_idx[word]
                w_sum = sen_sum - glb.base_vectors[w_index, :]
                if np.linalg.norm(w_sum) == 0.0:
                    continue
                else:
                    try:
                        vec_dict[word] += w_sum
                    except KeyError:
                        vec_dict[word] = w_sum
        return vec_dict

    def get_synonyms(self, word):
        probe = self.context_vectors[glb.word_to_idx[word], :]
        self.rank_words(np.dot(self.context_vectors, probe))

    def rank_words(self, comparison):
        rank = zip(range(len(glb.vocab)), comparison)
        rank = sorted(rank, key=operator.itemgetter(1), reverse=True)
        top_words = [(glb.idx_to_word[item[0]], item[1]) for item in rank[:10]]

        for word in top_words[:5]:
            print(word[0], word[1])

    def train(self, dim, vocab, batchsize=200):
        self.dim = dim

        glb.dim = dim
        glb.vocab = {w for w in vocab if w not in stopwords}

        glb.word_to_idx = {word: idx for idx, word in enumerate(glb.vocab)}
        glb.idx_to_word = {idx: word for word, idx in glb.word_to_idx.items()}

        glb.base_vectors = np.random.normal(loc=0, scale=1/(self.dim**0.5),
                                            size=(len(glb.vocab), self.dim))

        self.context_vectors = np.zeros((len(glb.vocab), self.dim))
        self.order_vectors = np.zeros((len(glb.vocab), self.dim))

        glb.pos_i = [self.unitary_vector() for i in range(5)]
        glb.neg_i = [self.unitary_vector() for i in range(5)]
        glb.pos_i = dict((i, j) for i, j in enumerate(glb.pos_i))
        glb.neg_i = dict((i, j) for i, j in enumerate(glb.neg_i))

        batch = []
        for article in self._corpus.articles:
            batch.append(article)
            if len(batch) == batchsize:
                cpus = mp.cpu_count()
                pool = mp.Pool(processes=cpus)
                result = pool.map_async(self.build_context_vectors, batch)
                for r in result.get():
                    for i, j in r.items():
                        self.context_vectors[glb.word_to_idx[i], :] += j
                pool.close()

                o_pool = mp.Pool(processes=cpus)
                result = o_pool.map_async(self.build_order_vectors, batch)
                for r in result.get():
                    for i, j in r.items():
                        self.order_vectors[glb.word_to_idx[i], :] += j
                pool.close()
                batch = []

        for word in glb.vocab:
            w_index = glb.word_to_idx[word]
            if np.all(self.context_vectors[w_index, :] == 0):
                self.context_vectors[w_index, :] = glb.base_vectors[w_index, :]

        norms = np.linalg.norm(self.context_vectors, axis=1)
        self.context_vectors = np.divide(self.context_vectors,
                                         norms[:, np.newaxis])

        norms = np.linalg.norm(self.order_vectors, axis=1)
        self.order_vectors = np.divide(self.order_vectors,
                                       norms[:, np.newaxis])

    def get_completions(self, word, position):
        v = self.order_vectors[glb.word_to_idx[word], :]
        if position > 0:
            probe = glb.deconvolve(glb.pos_i[position-1], v)
        if position < 0:
            probe = glb.deconvolve(glb.neg_i[abs(position+1)], v)
        self.rank_words(np.dot(glb.base_vectors, probe))

    def unit_vector(self):
        return self.normalize(np.random.normal(loc=0, size=1))

    def unitary_vector(self):
        v = np.random.normal(loc=0, scale=(1/(self.dim**0.5)), size=self.dim)
        fft_val = np.fft.fft(v)
        imag = fft_val.imag
        real = fft_val.real
        fft_norms = [np.sqrt(imag[n]**2 + real[n]**2) for n in range(len(v))]
        fft_unit = np.divide(fft_val, fft_norms)
        return np.fft.ifft(fft_unit).real


class SkipGram(EmbeddingModel):
    pass


class CBOW(EmbeddingModel):
    pass


class Word2Vec(SkipGram, CBOW):
    pass
