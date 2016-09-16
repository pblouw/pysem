import nltk

import spacy
import random

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from itertools import islice
from operator import itemgetter
from pysem.utils.multiprocessing import plainmap, PoolContainer
from pysem.utils.vsa import convolve, deconvolve, normalize, HRR

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en')


class RandomIndexing(object):
    """A base class for embedding models that make use of random indexing
    algorithms. These algorithms assign random unit vectors to all of the
    words in a vocabulary to be modelled. Then, operations on these unit
    vectors are performed while scanning through text to build up useful
    representations of the words in question.

    Specific random indexing models are subclasses of this class that specify
    a unique encoding algorithm to be used to train the model. Subclasses also
    specify model-specific methods for retrieving and manipulating learned
    embeddings in useful ways. The methods in the base class that implement
    training etc. make use of these specifications and thus only operate
    as intended when called via an instance of an appropriately defined
    subclass.

    Parameters:
    ----------
    corpus : datahandler object
        A text corpus to model. The provided object must implement a stream
        that yields corpus articles.
    vocab : list
        A list of strings that define the vocabulary of words to be modelled.

    Attributes:
    ----------
    corpus : datahandler object
        The text corpus to model using random indexing.
    vocab : list
        A list of strings that define the vocabulary of words to be modelled.
    vectors : numpy.ndarray
        An array whose rows correspond to the learned embeddings for each
        vocabulary item. Initialized as an array of zeros when the model is
        trained. The index of a word in the vocab is the same as the index of
        the row in the array containing the word's embedding.
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
        '''Convert an article into a list of lowercase sentences'''
        sen_list = tokenizer.tokenize(article)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(readonly.strip_num) for s in sen_list]
        sen_list = [s.translate(readonly.strip_pun) for s in sen_list]
        sen_list = [s.lower() for s in sen_list if len(s) > 5]
        sen_list = [nltk.word_tokenize(s) for s in sen_list]
        sen_list = [[w for w in s if w in readonly.vocab] for s in sen_list]
        return sen_list

    def run_pool(self, function, batch):
        '''Apply an embedding function to batch of articles in parallel'''
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map_async(function, batch)

            for encoding in results.get():
                for word, vector in encoding.items():
                    self.vectors[self.word_to_idx[word], :] += vector

    def train(self, dim, batchsize=100):
        '''Train a model with embeddings of the specified dimension'''
        self.dim = dim
        self.batchsize = batchsize
        self.vectors = np.zeros((len(self.vocab), dim))
        self._config_readonly()

        for batch in self._batches():
            sents = plainmap(self.preprocess, batch)
            self.run_pool(self.encode, sents)

        self.normalize()

    def normalize(self):
        '''Normalize the embeddings learned by the model to unit length'''
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1  # avoids divide by zero for words w/o embedding
        self.vectors = np.divide(self.vectors, norms[:, np.newaxis])

    def top_matches(self, probe, n, base_vectors=False):
        '''Return the top n embeddings matching a probe embedding'''
        if base_vectors:
            sims = np.dot(readonly.vectors, probe)
        else:
            sims = np.dot(self.vectors, probe)
        ranked = sorted(enumerate(sims), key=itemgetter(1), reverse=True)
        return [(self.idx_to_word[x[0]], x[1]) for x in ranked[:n]]

    def _config_readonly(self):
        '''Declares a PoolContainer as a global variable for shared
        memory used by all processes in a multiprocessing pool.'''
        global readonly
        readonly = PoolContainer(self.dim, self.vocab)

    def _batches(self):
        '''Use of islice requires that self.corpus.articles is a generator.'''
        while True:
            batch = list(islice(self.corpus.articles, self.batchsize))
            yield batch

            if len(batch) < self.batchsize:
                break

        self.corpus.reset_streams()


class ContextEmbedding(RandomIndexing):
    """A random indexing model that produces word embeddings on the basis of
    word co-occurences within sentences.
    """
    @staticmethod
    def encode(sen_list):
        '''Encode context info for each unique word in a list of sentences'''
        encodings = defaultdict(readonly.zeros)
        for sen in sen_list:
            sen_sum = sum([readonly[w].v for w in sen if w not in stopwords])
            for word in sen:
                word_sum = sen_sum - readonly[word].v
                encodings[word] += word_sum
        return encodings

    def get_nearest(self, word, n=5):
        '''Print the n nearest neighbors to a target word in context space'''
        probe = self.vectors[self.word_to_idx[word], :]
        top_n = self.top_matches(probe, n)
        for item in top_n:
            print(item[0], item[1])


class OrderEmbedding(RandomIndexing):
    """A random indexing model that produces word embeddings on the basis of
    positional word co-occurences within sentences.
    """
    @staticmethod
    def encode(sen_list):
        '''Encode order info for each unique word in a list of sentences'''
        encodings = defaultdict(readonly.zeros)
        win = readonly.winsize
        for sen in sen_list:
            for i in range(len(sen)):
                o_sum = HRR(readonly.zeros())
                for j in range(win):
                    if i+j+1 < len(sen):
                        w = readonly[sen[i+j+1]]
                        p = HRR(readonly.pos_idx[j])
                        o_sum += w * p
                    if i-j-1 >= 0:
                        w = readonly[sen[i-j-1]]
                        p = HRR(readonly.neg_idx[j])
                        o_sum += w * p
                encodings[sen[i]] += o_sum.v
        return encodings

    def get_completions(self, word, position, n=5):
        '''Print the n most likely words to occur in the specified position
        relative to the provided target word in order space'''
        embedding = self.vectors[self.word_to_idx[word], :]
        if position > 0:
            probe = deconvolve(readonly.pos_idx[position-1], embedding)
        else:
            probe = deconvolve(readonly.neg_idx[abs(position+1)], embedding)

        top_n = self.top_matches(probe, n, base_vectors=True)
        for item in top_n:
            print(item[0], item[1])

    def get_resonants(self, phrase, n=5):
        '''Print the n most likely words to occur in the specified position
        relative to the provided phrase in order space'''
        probe = self.get_vector_encoding(phrase)
        top_n = self.top_matches(probe, n)
        for item in top_n:
            print(item[0], item[1])

    def get_vector_encoding(self, phrase):
        '''Converts a phrase query into a vector encoding the linguistic
        context surrounding a target position'''
        words = phrase.split()
        index = words.index('__')
        probe = np.zeros(self.dim)
        for word in words:
            if word == '__':
                continue
            w = readonly[word].v
            if words.index(word) < index:
                p = readonly.neg_idx[index-words.index(word)-1]
                probe += convolve(w, p)
            if words.index(word) > index:
                p = readonly.pos_idx[words.index(word)-index-1]
                probe += convolve(w, p)

        return normalize(probe)

    def _config_readonly(self):
        '''Configures PoolContainer to include vectors that tag different
        positional contexts around a target word'''
        global readonly
        readonly = PoolContainer(self.dim, self.vocab)
        readonly.build_position_tags()


class SyntaxEmbedding(RandomIndexing):
    """A random indexing model that produces word embeddings on the basis of
    syntactic dependency relations within sentences.
    """
    @staticmethod
    def encode(article):
        doc = nlp(article)
        encodings = defaultdict(readonly.zeros)
        targpos = 'VERB'

        for sen in doc.sents:
            for token in sen:
                word = token.lower_
                if word in readonly.vocab and token.pos_ == targpos:
                    for c in token.children:
                        if c.dep_ in readonly.verb_deps:
                            role = readonly.verb_deps[c.dep_]
                            if c.lower_ in readonly.vocab:
                                filler = readonly[c.lower_].v
                                binding = convolve(role, filler)
                                encodings[word] += binding
        return encodings

    def _config_readonly(self):
        '''Configures PoolContainer to include vectors that tag different
        syntactic neighbors of a target word'''
        global readonly
        readonly = PoolContainer(self.dim, self.vocab)
        readonly.build_dependency_tags()

    def train(self, dim, batchsize=100):
        '''Overrides usual RandomIndexing train method to run multiprocessing
        pools on batches of articles instead of batches of sentences'''
        self.dim = dim
        self.batchsize = batchsize
        self.vectors = np.zeros((len(self.vocab), dim))
        self._config_readonly()

        for batch in self._batches():
            self.run_pool(self.encode, batch)

        self.normalize()

    def get_verb_neighbors(self, word, dep, n=5):
        '''Print the n most likely words to occupy the specified syntactic
        dependency relative to the provided target word in syntax space'''
        embedding = self.vectors[self.word_to_idx[word], :]
        probe = deconvolve(readonly.verb_deps[dep], embedding)
        top_n = self.top_matches(probe, n, base_vectors=True)
        for item in top_n:
            print(item[0], item[1])
