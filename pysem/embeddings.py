import nltk
import operator
import spacy
import random

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from itertools import islice
from pysem.utils.multiprocessing import plainmap, PoolData
from pysem.utils.vsa import convolve, deconvolve, HRR

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en')


class RandomIndexing(object):
    """A base class for embedding models that make use of random indexing
    algorithms. These algorithms assign random unit vectors to all of the
    words in a vocabulary to be modelled. Then, operations on these unit
    vectors are performed while scanning through text to build up useful
    representations of the vocabulary items.

    Parameters:
    ----------
    corpus : datahandler object
        A text corpus to model. The provided object must implement a stream
        that yields corpus articles.
    vocab : list
        A list of strings that define the vocabulary of words to be modelled.

    Attributes:
    ----------
    vocab : list
        A list of strings that define the vocabulary of words to be modelled.
    vectors : numpy.ndarray
        An array whose rows correspond to the learned embeddings for each
        vocabulary item. Initialized as an array of zeros.
    """
    def __init__(self, corpus, vocab):
        self.corpus = corpus
        self.vocab = vocab

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, wordset):
        randitem = random.choice(wordset)
        if isinstance(wordset, list) and isinstance(randitem, str):
            self._vocab = wordset
            self.word_to_idx = {word: idx for idx, word in enumerate(wordset)}
            self.idx_to_word = {idx: word for idx, word in enumerate(wordset)}
        else:
            raise TypeError('The vocabulary must be a list of strings')

    @staticmethod
    def encode():
        raise NotImplementedError('Random indexing models must implement a \
                                   staticmethod for encoding text corpora')

    @staticmethod
    def preprocess(article):
        sen_list = tokenizer.tokenize(article)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(readonly.strip_num) for s in sen_list]
        sen_list = [s.translate(readonly.strip_pun) for s in sen_list]
        sen_list = [s.lower() for s in sen_list if len(s) > 5]
        sen_list = [nltk.word_tokenize(s) for s in sen_list]
        sen_list = [[w for w in s if w in readonly.vocab] for s in sen_list]
        return sen_list

    def normalize(self):
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1
        self.vectors = np.divide(self.vectors, norms[:, np.newaxis])

    def run_pool(self, function, batch):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map_async(function, batch)

            for encoding in results.get():
                for word, vector in encoding.items():
                    self.vectors[self.word_to_idx[word], :] += vector

    def train(self, dim, batchsize=100):
        self.dim = dim
        self.batchsize = batchsize
        self.vectors = np.zeros((len(self.vocab), dim))

        global readonly
        readonly = PoolData(dim, self.vocab)

        for batch in self.batches():
            sents = plainmap(self.preprocess, batch)
            self.run_pool(self.encode, sents)

        self.normalize()

    def rank_words(self, dotproducts, n=5):
        scores = zip(range(len(self.vocab)), dotproducts)
        ranked = sorted(scores, key=operator.itemgetter(1), reverse=True)
        top_n = [(self.idx_to_word[x[0]], x[1]) for x in ranked[:n]]
        for pair in top_n:
            print(pair[0], pair[1])

    def get_nearest(self, word):
        probe = self.vectors[self.word_to_idx[word], :]
        self.rank_words(np.dot(self.vectors, probe))

    def batches(self):
        while True:
            batch = list(islice(self.corpus.articles, self.batchsize))
            yield batch

            if len(batch) < self.batchsize:
                break

        self.corpus.reset_streams()


class ContextEmbedding(RandomIndexing):

    @staticmethod
    def encode(sen_list):
        encodings = defaultdict(readonly.zeros)
        for sen in sen_list:
            sen_sum = sum([readonly[w].v for w in sen if w not in stopwords])
            for word in sen:
                word_sum = sen_sum - readonly[word].v
                encodings[word] += word_sum
        return encodings


class OrderEmbedding(RandomIndexing):

    @staticmethod
    def encode(sen_list):
        encodings = defaultdict(readonly.zeros)
        win = 5
        for sen in sen_list:
            for x in range(len(sen)):
                o_sum = HRR(readonly.zeros())
                for y in range(win):
                    if x+y+1 < len(sen):
                        w = readonly[sen[x+y+1]]
                        p = HRR(readonly.pos_i[y])
                        o_sum += w * p
                    if x-y-1 >= 0:
                        w = readonly[sen[x-y-1]]
                        p = HRR(readonly.neg_i[y])
                        o_sum += w * p
                encodings[sen[x]] += o_sum.v
        return encodings

    def get_completions(self, word, position):
        v = self.vectors[self.word_to_idx[word], :]
        if position > 0:
            probe = deconvolve(readonly.pos_i[position-1], v)
        if position < 0:
            probe = deconvolve(readonly.neg_i[abs(position+1)], v)
        self.rank_words(np.dot(readonly.vectors, probe))


class SyntaxEmbedding(RandomIndexing):

    @staticmethod
    def encode(article):
        doc = nlp(article)
        encodings = defaultdict(readonly.zeros)

        for sent in doc.sents:
            for token in sent:
                word = token.orth_.lower()
                if word in readonly.vocab:
                    pos = token.pos_

                    children = [c for c in token.children]
                    for c in children:
                        dep = c.dep_

                        if pos == 'VERB' and dep in readonly.verb_deps:
                            role = readonly.verb_deps[dep]
                            orth = c.orth_.lower()
                            if orth in readonly.vocab:
                                filler = readonly[orth].v
                                binding = convolve(role, filler)
                                encodings[word] += binding
        return encodings

    def train(self, dim, batchsize=100):
        self.dim = dim
        self.batchsize = batchsize
        self.vectors = np.zeros((len(self.vocab), dim))

        for batch in self.batches():
            self.run_pool(self.encode, batch)

        self.normalize()

    def get_verb_neighbor(self, word, dep):
        v = self.vectors[self.word_to_idx[word], :]
        probe = deconvolve(readonly.verb_deps[dep], v)
        self.rank_words(np.dot(readonly.vectors, probe))


# class CombinedEmbedding(RandomIndexing):

#     def train(dim, batchsize=100):
#         self.context_model = ContextEmbedding()
