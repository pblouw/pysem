import nltk
import operator
import spacy

import numpy as np
import multiprocessing as mp

from collections import defaultdict
from pysem.hrr import Vocabulary, HRR
from pysem.mputils import apply_async

tokenizer = nltk.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')

vocab = None

nlp = spacy.load('en')

# deps = dict()
# deps['VERB'] = set(['nsubj', 'dobj'])


def zeros():
    return np.zeros(vocab.dimensions)


class EmbeddingModel(object):
    """Base class for embedding models"""
    def __init__(self):
        pass

    def rank_words(self, dotproducts, n=5):
        scores = zip(range(len(self.wordlist)), dotproducts)
        ranked = sorted(scores, key=operator.itemgetter(1), reverse=True)
        top_n = [(self.index_to_word[x[0]], x[1]) for x in ranked[:n]]
        for pair in top_n:
            print(pair[0], pair[1])

    def get_nearest(self, word):
        probe = self.context_vectors[self.word_to_index[word], :]
        self.rank_words(np.dot(self.context_vectors, probe))

    def get_completions(self, word, position):
        v = self.order_vectors[self.word_to_index[word], :]
        if position > 0:
            probe = vocab.deconvolve(vocab.pos_i[position-1], v)
        if position < 0:
            probe = vocab.deconvolve(vocab.neg_i[abs(position+1)], v)
        self.rank_words(np.dot(vocab.vectors, probe))

    def get_verb_neighbor(self, word, dep):
        v = self.syntax_vectors[self.word_to_index[word], :]
        probe = vocab.deconvolve(vocab.verb_deps[dep], v)
        self.rank_words(np.dot(vocab.vectors, probe))


class RandomIndexing(EmbeddingModel):

    def __init__(self, corpus):
        self._corpus = corpus
        self.cpus = mp.cpu_count()
        self.lookup = {'context': self._update_context_vectors,
                       'order': self._update_order_vectors,
                       'syntax': self._update_syntax_vectors}

    @staticmethod
    def _preprocess(article):
        sen_list = tokenizer.tokenize(article)
        sen_list = [s.replace('\n', ' ') for s in sen_list]
        sen_list = [s.translate(vocab.strip_num) for s in sen_list]
        sen_list = [s.translate(vocab.strip_pun) for s in sen_list]
        sen_list = [s.lower() for s in sen_list if len(s) > 5]
        sen_list = [nltk.word_tokenize(s) for s in sen_list]
        sen_list = [[w for w in s if w in vocab.wordlist] for s in sen_list]
        return sen_list

    @staticmethod
    def _encode_context(sen_list):
        encodings = defaultdict(zeros)
        for sen in sen_list:
            sen_sum = sum([vocab[w].v for w in sen if w not in stopwords])
            for word in sen:
                word_sum = sen_sum - vocab[word].v
                encodings[word] += word_sum
        return encodings

    @staticmethod
    def _encode_order(sen_list):
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
    def _encode_syntax(article):
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
                            # print(c.orth_, dep, word)
                            role = vocab.verb_deps[dep]
                            orth = c.orth_.lower()
                            if orth in vocab.wordlist:
                                filler = vocab[orth].v
                                binding = vocab.convolve(role, filler)
                                encodings[word] += binding
        return encodings

    def _encode_flagged(self, batch):
        for flag in self.flags:
            self.lookup[flag](batch)

    def _encode_all(self, batch):
        sents = apply_async(self._preprocess, batch)
        sents = [lst for lst in sents if len(lst) > 1]
        self._run_pool(self._encode_syntax, batch, self.syntax_vectors)
        self._run_pool(self._encode_context, sents, self.context_vectors)
        self._run_pool(self._encode_order, sents, self.order_vectors)

    def _normalize_encoding(self, encoding):
        for word in vocab.wordlist:
            index = vocab.word_to_index[word]
            if np.all(encoding[index, :] == 0):
                encoding[index, :] = vocab[word].v

        norms = np.linalg.norm(encoding, axis=1)
        encoding = np.divide(encoding, norms[:, np.newaxis])
        return encoding

    def _update_context_vectors(self, batch):
        sents = apply_async(self._preprocess, batch)
        self._run_pool(self._encode_context, sents, self.context_vectors)

    def _update_order_vectors(self, batch):
        sents = apply_async(self._preprocess, batch)
        self._run_pool(self._encode_order, sents, self.order_vectors)

    def _update_syntax_vectors(self, batch):
        self._run_pool(self._encode_syntax, batch, self.syntax_vectors)

    def _batches(self):
        batch = []
        for article in self._corpus.articles:
            batch.append(article)
            if len(batch) % self.batchsize == 0 and len(batch) > 0:
                yield batch
                batch = []
        yield batch  # collects leftover articles in a batch < batchsize

    def _run_pool(self, function, batch, encoding):
        with mp.Pool(processes=self.cpus) as pool:
            result = pool.map_async(function, batch)
            for _ in result.get():
                for word, vec in _.items():
                    encoding[vocab.word_to_index[word], :] += vec

    def train(self, dim, wordlist, flags=None, batchsize=500):
        self.dim = dim
        self.wordlist = wordlist
        self.flags = flags
        self.batchsize = batchsize
        self.word_to_index = {word: idx for idx, word in enumerate(wordlist)}
        self.index_to_word = {idx: word for idx, word in enumerate(wordlist)}

        global vocab
        vocab = Vocabulary(dim, wordlist)

        self.syntax_vectors = np.zeros((len(wordlist), self.dim))
        self.order_vectors = np.zeros((len(wordlist), self.dim))
        self.context_vectors = np.zeros((len(wordlist), self.dim))

        for batch in self._batches():
            if flags:
                self._encode_flagged(batch)
            else:
                self._encode_all(batch)

        self.context_vectors = self._normalize_encoding(self.context_vectors)
        self.order_vectors = self._normalize_encoding(self.order_vectors)
        self.syntax_vectors = self._normalize_encoding(self.syntax_vectors)


class SkipGram(EmbeddingModel):
    pass


class CBOW(EmbeddingModel):
    pass


class Word2Vec(SkipGram, CBOW):
    pass
