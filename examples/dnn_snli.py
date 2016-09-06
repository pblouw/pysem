import platform
import random
import time
import sys
import numpy as np

from pysem.corpora import SNLI
from pysem.networks import DependencyNetwork
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
pretrained = 'pretrained_snli_embeddings.pickle'

encoder = DependencyNetwork(dim=dim, vocab=snli.vocab, pretrained=pretrained)
classifier = MultiLayerPerceptron(di=2*dim, dh=dim, do=3)

start_time = time.time()

model = CompositeModel(snli, encoder, classifier)

model.train(iters=10, bsize=100, rate=0.01, acc_interval=10)
model.acc.append(model.dnn_accuracy())

print('Total runtime: ', time.time() - start_time)

model.plot()