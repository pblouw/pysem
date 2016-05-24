import nltk
import string
import operator
import spacy

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from hrr import Vocabulary, HRR

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')

shared = None

nlp = spacy.load('en')

deps = dict()
deps['VERB'] = set(['nsubj', 'dobj'])


def zeros():
    return np.zeros(shared.dimensions)


class EmbeddingModel(object):
    """Base class for embedding models"""
    def __init__(self):
        pass


class RandomIndexing(EmbeddingModel):

    def __init__(self, corpus):
        self._corpus = corpus

    @staticmethod
    def encode(article):
        sen_list = tokenizer.tokenize(article)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(shared.strip_num) for s in sen_list]
        sen_list = [s.translate(shared.strip_pun) for s in sen_list]
        sen_list = [s.lower() for s in sen_list if len(s) > 5]
        sen_list = [nltk.word_tokenize(s) for s in sen_list]
        sen_list = [[w for w in s if w in shared.wordlist] for s in sen_list]

        ord_dict = defaultdict(zeros)
        con_dict = defaultdict(zeros)

        maxn = 5
        for sen in sen_list:
            sen_sum = sum([shared[w].v for w in sen if w not in stopwords])

            for x in range(len(sen)):
                o_sum = HRR(np.zeros(shared.dimensions))
                for y in range(maxn):
                    if x+y+1 < len(sen):
                        w = shared[sen[x+y+1]]
                        p = HRR(shared.pos_i[y])
                        o_sum += w * p
                    if x-y-1 >= 0:
                        w = shared[sen[x-y-1]]
                        p = HRR(shared.neg_i[y])
                        o_sum += w * p

                word = sen[x]
                w_sum = sen_sum - shared[word].v

                ord_dict[word] += o_sum.v
                con_dict[word] += w_sum

        return (con_dict, ord_dict)

    @staticmethod
    def syntax(article):
        doc = nlp(article)
        syn_dict = dict()

        for sent in doc.sents:
            for token in sent:
                word = token.orth_.lower()
                if word in shared.wordlist:
                    pos = token.pos_

                    children = [c for c in token.children]
                    for c in children:
                        dep = c.dep_

                        if pos == 'VERB' and dep in shared.verb_deps:
                            print(c.orth_, dep, word)
                            role = shared.verb_deps[dep]
                            orth = c.orth_.lower()
                            if orth in shared.wordlist:
                                filler = shared[orth].v
                                binding = shared.convolve(role, filler)
                                try:
                                    syn_dict[word] += binding
                                except KeyError:
                                    syn_dict[word] = binding

        return syn_dict

    def get_synonyms(self, word):
        probe = self.context_vectors[shared.word_to_index[word], :]
        self.rank_words(np.dot(self.context_vectors, probe))

    def rank_words(self, comparison):
        rank = zip(range(len(shared.wordlist)), comparison)
        rank = sorted(rank, key=operator.itemgetter(1), reverse=True)
        top_words = [(shared.index_to_word[item[0]], item[1])
                     for item in rank[:10]]

        for word in top_words[:5]:
            print(word[0], word[1])

    def get_completions(self, word, position):
        v = self.order_vectors[shared.word_to_index[word], :]
        if position > 0:
            probe = shared.deconvolve(shared.pos_i[position-1], v)
        if position < 0:
            probe = shared.deconvolve(shared.neg_i[abs(position+1)], v)
        self.rank_words(np.dot(shared.vectors, probe))

    def get_verb_neighbor(self, word, dep):
        v = self.dep_vectors[shared.word_to_index[word], :]
        probe = shared.deconvolve(shared.verb_deps[dep], v)
        self.rank_words(np.dot(shared.vectors, probe))

    def train(self, dim, vocab, batchsize=500):
        self.dim = dim
        self.cpus = mp.cpu_count()

        global shared
        shared = Vocabulary(dim, vocab)
        shared.strip_pun = str.maketrans({key: None
                                          for key in string.punctuation})
        shared.strip_num = str.maketrans({key: None for key in string.digits})

        self.context_vectors = np.zeros((len(vocab), self.dim))
        self.order_vectors = np.zeros((len(vocab), self.dim))
        self.dep_vectors = np.zeros((len(vocab), self.dim))

        batch = []
        for article in self._corpus.articles:
            batch.append(article)
            if len(batch) % batchsize == 0 and len(batch) > 0:
                self.run_pool(self.encode, batch)
                self.run_pool(self.syntax, batch, syntax=True)
                batch = []

        for word in shared.wordlist:
            w_index = shared.word_to_index[word]
            if np.all(self.context_vectors[w_index, :] == 0):
                self.context_vectors[w_index, :] = shared[word].v
            if np.all(self.order_vectors[w_index, :] == 0):
                self.order_vectors[w_index, :] = shared[word].v
            if np.all(self.dep_vectors[w_index, :] == 0):
                self.dep_vectors[w_index, :] = shared[word].v

        norms = np.linalg.norm(self.context_vectors, axis=1)
        self.context_vectors = np.divide(self.context_vectors,
                                         norms[:, np.newaxis])

        norms = np.linalg.norm(self.order_vectors, axis=1)
        self.order_vectors = np.divide(self.order_vectors,
                                       norms[:, np.newaxis])

        norms = np.linalg.norm(self.dep_vectors, axis=1)
        self.dep_vectors = np.divide(self.dep_vectors,
                                     norms[:, np.newaxis])

    def run_pool(self, function, batch, syntax=False):
        with mp.Pool(processes=self.cpus) as pool:
            result = pool.map_async(function, batch)
            for r in result.get():
                if syntax:
                    for i, j in r.items():
                        self.dep_vectors[shared.word_to_index[i], :] += j
                    continue

                context = r[0]
                order = r[1]

                for i, j in order.items():
                    self.order_vectors[shared.word_to_index[i], :] += j

                for i, j in context.items():
                    self.context_vectors[shared.word_to_index[i], :] += j


class SkipGram(EmbeddingModel):
    pass


class CBOW(EmbeddingModel):
    pass


class Word2Vec(SkipGram, CBOW):
    pass
