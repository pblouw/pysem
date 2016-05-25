import string
import numpy as np


class Vocabulary(object):
    ''''''
    def __init__(self, dimensions, wordlist):
        self.wordlist = wordlist
        self.dimensions = dimensions
        self.vectors = np.random.normal(loc=0, scale=1/dimensions**0.5,
                                        size=(len(wordlist), dimensions))

        self.word_to_index = {word: idx for idx, word in enumerate(wordlist)}
        self.index_to_word = {idx: word for idx, word in enumerate(wordlist)}
        self.build_tags()

        self.strip_pun = str.maketrans({x: None for x in string.punctuation})
        self.strip_num = str.maketrans({x: None for x in string.digits})

    def __getitem__(self, word):
        index = self.word_to_index[word]
        return HRR(self.vectors[index, :])

    def convolve(self, a, b):
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

    def deconvolve(self, a, b):
        return self.convolve(np.roll(a[::-1], 1), b)

    def normalize(self, v):
        if self.norm_of_vector(v) > 0:
            return v / self.norm_of_vector(v)

    def norm_of_vector(self, v):
        return np.linalg.norm(v)

    def unitary_vector(self):
        dim = self.dimensions
        v = np.random.normal(loc=0, scale=(1/(dim**0.5)), size=dim)
        fft_val = np.fft.fft(v)
        imag = fft_val.imag
        real = fft_val.real
        fft_norms = [np.sqrt(imag[n]**2 + real[n]**2) for n in range(len(v))]
        fft_unit = np.divide(fft_val, fft_norms)
        return np.fft.ifft(fft_unit).real

    def build_tags(self):
        deps = set(['nsubj', 'dobj'])
        self.pos_i = [self.unitary_vector() for i in range(5)]
        self.neg_i = [self.unitary_vector() for i in range(5)]
        self.pos_i = dict((i, j) for i, j in enumerate(self.pos_i))
        self.neg_i = dict((i, j) for i, j in enumerate(self.neg_i))
        self.verb_deps = {dep: self.unitary_vector() for dep in deps}


class HRR(object):

    def __init__(self, vector, unitary=False):
        # self.v = self.normalize(vector)
        self.v = vector
        if unitary:
            self.make_unitary()

    def __sub__(self, other):
        if isinstance(other, HRR):
            return HRR(self.v - other.v)
        else:
            raise Exception('Both objects must by HRRs')

    def __add__(self, other):
        if isinstance(other, HRR):
            return HRR(self.v + other.v)
        else:
            raise Exception('Both objects must by HRRs')

    def __mul__(self, other):
        if isinstance(other, HRR):
            return self.convolve(other)
        else:
            raise Exception('Both objects must by HRRs')

    def __invert__(self):
        return HRR(np.roll(self.v[::-1], 1))

    def normalize(self, v):
        if self.norm_of_vector(v) > 0:
            return v / self.norm_of_vector(v)
        else:
            return np.zeros(len(v))

    def norm_of_vector(self, v):
        return np.linalg.norm(v)

    def convolve(self, other):
        v = np.fft.ifft(np.fft.fft(self.v) * np.fft.fft(other.v)).real
        return HRR(v)

    def make_unitary(self):
        '''Make the HRR unitary'''
        fft_val = np.fft.fft(self.v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        fft_unit = fft_val / fft_norms
        self.v = np.array((np.fft.ifft(fft_unit)).real)
