import platform
import random
import time
import sys
import numpy as np

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork
from pysem.utils.ml import MultiLayerPerceptron
from pysem.utils.snli import CompositeModel
from itertools import islice

if platform.system() == 'Linux':
    snlipath = '/home/pblouw/corpora/snli_1.0/'
    wikipath = '/home/pblouw/corpora/wikipedia'
    cachepath = '/home/pblouw/cache/'
else:
    snlipath = '/Users/peterblouw/corpora/snli_1.0/'
    wikipath = '/Users/peterblouw/corpora/wikipedia'
    cachepath = '/Users/peterblouw/cache/'


snli = SNLI(snlipath)
snli.extractor = snli.get_xy_pairs
snli.load_vocab('snli_words.pickle')

dim = 300

encoder = RecurrentNetwork(dim=dim, vocab=snli.vocab)
classifier = MultiLayerPerceptron(di=2*dim, dh=dim, do=3)

model = CompositeModel(snli, encoder, classifier)


start_time = time.time()

model.train(epochs=0.001, bsize=100, rate=0.01, acc_interval=2)
print(model.rnn_accuracy())
model.plot()

print('Total runtime: ', time.time() - start_time)
