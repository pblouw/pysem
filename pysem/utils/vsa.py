import numpy as np


def convolve(v1, v2):
    return np.fft.ifft(np.fft.fft(v1) * np.fft.fft(v2)).real


def deconvolve(v1, v2):
    return convolve(np.roll(v1[::-1], 1), v2)


def normalize(v):
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    else:
        return v


def unitary_vector(dim):
    v = np.random.normal(loc=0, scale=(1/(dim**0.5)), size=dim)
    fft_val = np.fft.fft(v)
    imag = fft_val.imag
    real = fft_val.real
    fft_norms = [np.sqrt(imag[n]**2 + real[n]**2) for n in range(len(v))]
    fft_unit = np.divide(fft_val, fft_norms)
    return np.fft.ifft(fft_unit).real


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
