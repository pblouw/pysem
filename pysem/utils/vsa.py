import numpy as np


def convolve(v1, v2):
    '''Compute the circular convolution of two vectors'''
    return np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2)).real


def deconvolve(v1, v2):
    '''Compute the circular correlation of the first vector with the second'''
    return convolve(np.roll(v1[::-1], 1), v2)


def normalize(v):
    '''Normalize a vector to unit length'''
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    else:
        return v


def unitary_vector(dim):
    '''Produce a vector whose approximate and exact inverses are equivalent'''
    v = np.random.normal(loc=0, scale=(1/(dim**0.5)), size=dim)
    fft_val = np.fft.fft(v)
    imag = fft_val.imag
    real = fft_val.real
    fft_norms = [np.sqrt(imag[n]**2 + real[n]**2) for n in range(len(v))]
    fft_unit = np.divide(fft_val, fft_norms)
    return np.fft.ifft(fft_unit).real


def get_convolution_matrix(hrr):
    '''Produce a matrix that applies a linear transform equivalent to
    convolution by the supplied hrr'''
    d = len(hrr)
    t = []
    for i in range(d):
        t.append([hrr[(i - j) % d] for j in range(d)])
    return np.array(t)


class HRR(object):
    """A holographic reduced representation, as defined by Plate (2003)."""
    def __init__(self, vector, unitary=False):
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

    def normalize(self):
        if self.norm_of_vector() > 0:
            self.v = self.v / self.norm_of_vector()
        else:
            self.v = np.zeros_like(self.v)

    def norm_of_vector(self):
        return np.linalg.norm(self.v)

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
