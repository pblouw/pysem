import numpy as np


class Vocabulary(object):
    ''''''
    def __init__(self, dimensions, wordlist):
        self.wordlist = wordlist
        self.dimensions = dimensions
        self.vectors = np.random.normal(loc=0, scale=1/dimensions**0.5,
                                        size=(len(wordlist), dimensions))

        self.word_to_index = {word: idx for idx, word in enumerate(wordlist)}
        self.index_to_word = {idx: word for word, idx
                              in self.word_to_index.items()}

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


class HRR(object):
    """ A holographic reduced representation, based on the work of Tony Plate
    (2003). HRRs constitute a vector symbolic architecture for encoding
    symbolic structures in high-dimensional vector spaces. HRRs utilize
    circular convolution for binding and vector addition for superposition.
    Binding and unbinding are approximate, not exact, so HRRs are best thought
    of as providing a lossy compression of a symbol structure into a vector
    space.

    Parameters
    -----------
    data : int or np.array
        An int specifies the dimensionality of a randomly generated HRR. An
        np.array can provided instead to specify the value on each dimension of
        the HRR. (note that these values must be statistically distributed in
        a particular way in order for the HRR to act as expected)
    unitary : bool, optional
        If True, the generated HRR will be unitary, i.e. its exact and
        approximate inverses are equivalent.
    """
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
