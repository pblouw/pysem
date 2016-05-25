import nltk
import operator
import spacy

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from hrr import Vocabulary, HRR
from mputils import apply_async

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')

vocab = None

nlp = spacy.load('en')

deps = dict()
deps['VERB'] = set(['nsubj', 'dobj'])


def zeros():
    return np.zeros(vocab.dimensions)


class EmbeddingModel(object):
    """Base class for embedding models"""
    def __init__(self):
        pass


class RandomIndexing(EmbeddingModel):

    def __init__(self, corpus):
        self._corpus = corpus
        self.cpus = mp.cpu_count()

    @staticmethod
    def preprocess(article):
        sen_list = tokenizer.tokenize(article)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(vocab.strip_num) for s in sen_list]
        sen_list = [s.translate(vocab.strip_pun) for s in sen_list]
        sen_list = [s.lower() for s in sen_list if len(s) > 5]
        sen_list = [nltk.word_tokenize(s) for s in sen_list]
        sen_list = [[w for w in s if w in vocab.wordlist] for s in sen_list]
        return sen_list

    @staticmethod
    def encode_context(sen_list):
        encodings = defaultdict(zeros)
        for sen in sen_list:
            sen_sum = sum([vocab[w].v for w in sen if w not in stopwords])
            for word in sen:
                word_sum = sen_sum - vocab[word].v
                encodings[word] += word_sum
        return encodings

    @staticmethod
    def encode_order(sen_list):
        encodings = defaultdict(zeros)
        win = 5
        for sen in sen_list:
            for x in range(len(sen)):
                o_sum = HRR(zeros())
                for y in range(win):
                    if x+y+1 < len(sen):
                        w = vocab[sen[x+y+1]]
                        p = HRR(vocab.pos_i[y])
                        o_sum += w * p
                    if x-y-1 >= 0:
                        w = vocab[sen[x-y-1]]
                        p = HRR(vocab.neg_i[y])
                        o_sum += w * p
                encodings[sen[x]] += o_sum.v
        return encodings

    @staticmethod
    def encode_syntax(article):
        doc = nlp(article)
        encodings = defaultdict(zeros)

        for sent in doc.sents:
            for token in sent:
                word = token.orth_.lower()
                if word in vocab.wordlist:
                    pos = token.pos_

                    children = [c for c in token.children]
                    for c in children:
                        dep = c.dep_

                        if pos == 'VERB' and dep in vocab.verb_deps:
                            print(c.orth_, dep, word)
                            role = vocab.verb_deps[dep]
                            orth = c.orth_.lower()
                            if orth in vocab.wordlist:
                                filler = vocab[orth].v
                                binding = vocab.convolve(role, filler)
                                encodings[word] += binding
        return encodings

    def get_synonyms(self, word):
        probe = self.context_vectors[vocab.word_to_index[word], :]
        self.rank_words(np.dot(self.context_vectors, probe))

    def rank_words(self, comparison):
        rank = zip(range(len(vocab.wordlist)), comparison)
        rank = sorted(rank, key=operator.itemgetter(1), reverse=True)
        top_words = [(vocab.index_to_word[item[0]], item[1])
                     for item in rank[:10]]

        for word in top_words[:5]:
            print(word[0], word[1])

    def get_completions(self, word, position):
        v = self.order_vectors[vocab.word_to_index[word], :]
        if position > 0:
            probe = vocab.deconvolve(vocab.pos_i[position-1], v)
        if position < 0:
            probe = vocab.deconvolve(vocab.neg_i[abs(position+1)], v)
        self.rank_words(np.dot(vocab.vectors, probe))

    def get_verb_neighbor(self, word, dep):
        v = self.dep_vectors[vocab.word_to_index[word], :]
        probe = vocab.deconvolve(vocab.verb_deps[dep], v)
        self.rank_words(np.dot(vocab.vectors, probe))

    def normalize_encoding(self, encoding):
        for word in vocab.wordlist:
            index = vocab.word_to_index[word]
            if np.all(encoding[index, :] == 0):
                encoding[index, :] = vocab[word].v

        norms = np.linalg.norm(encoding, axis=1)
        encoding = np.divide(encoding, norms[:, np.newaxis])
        return encoding

    def train(self, dim, wordlist, batchsize=500):
        self.dim = dim

        global vocab
        vocab = Vocabulary(dim, wordlist)

        self.context_vectors = np.zeros((len(wordlist), self.dim))
        self.order_vectors = np.zeros((len(wordlist), self.dim))
        self.dep_vectors = np.zeros((len(wordlist), self.dim))

        batch = []
        for article in self._corpus.articles:
            batch.append(article)
            if len(batch) % batchsize == 0 and len(batch) > 0:
                self.process_batch(batch)
                batch = []

        self.process_batch(batch)

        self.context_vectors = self.normalize_encoding(self.context_vectors)
        self.order_vectors = self.normalize_encoding(self.order_vectors)
        self.dep_vectors = self.normalize_encoding(self.dep_vectors)

    def process_batch(self, batch):
        sents = apply_async(self.preprocess, batch)
        sents = [lst for lst in sents if len(lst) > 1]
        self.run_pool(self.encode_syntax, batch, self.dep_vectors)
        self.run_pool(self.encode_context, sents, self.context_vectors)
        self.run_pool(self.encode_order, sents, self.order_vectors)

    def run_pool(self, function, batch, encoding):
        with mp.Pool(processes=self.cpus) as pool:
            result = pool.map_async(function, batch)
            for _ in result.get():
                for word, vec in _.items():
                    encoding[vocab.word_to_index[word], :] += vec


class SkipGram(EmbeddingModel):
    pass


class CBOW(EmbeddingModel):
    pass


class Word2Vec(SkipGram, CBOW):
    pass
