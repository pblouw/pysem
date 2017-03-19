import platform
import random
import time
import sys
import pickle
import numpy as np

from pysem.corpora import SNLI
from pysem.networks import DependencyNetwork
from pysem.utils.ml import MultiLayerPerceptron
from pysem.utils.snli import CompositeModel
from itertools import islice

snli = SNLI('/home/pblouw/snli_1.0/')
snli.load_vocab('snli_words.pickle')

dim = 300
pretrained = 'pretrained_snli_embeddings.pickle'

encoder = DependencyNetwork(dim=dim, vocab=snli.vocab, pretrained=pretrained)
classifier = MultiLayerPerceptron(di=2*dim, dh=dim, do=3)

model = CompositeModel(snli, encoder, classifier)

start_time = time.time()

model = CompositeModel(snli, encoder, classifier)

model.train(iters=50000, bsize=100, rate=0.005, log_interval=1000, schedule=12000)
print(model.dnn_accuracy(model.test_data))
model.plot()

print('Total runtime: ', time.time() - start_time)

with open('dnn_model', 'wb') as pfile:
	pickle.dump(model, pfile)

with open('dnn_model', 'rb') as pfile:
	test_model = pickle.load(pfile)

print('Testing Model Loading')
print(test_model.dnn_accuracy(test_model.test_data))
