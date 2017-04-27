import time
import pickle

from pysem.corpora import SNLI
from pysem.networks import DependencyNetwork
from pysem.utils.ml import MultiLayerPerceptron
from pysem.utils.snli import CompositeModel

snli = SNLI('/Users/peterblouw/corpora/snli_1.0/')
snli.load_vocab('snli_words.pickle')

dim = 300
pretrained = 'pretrained_snli_embeddings.pickle'

encoder = DependencyNetwork(dim=dim, vocab=snli.vocab, pretrained=pretrained)
classifier = MultiLayerPerceptron(di=2*dim, dh=dim, do=3)

start_time = time.time()
model = CompositeModel(snli, encoder, classifier)
model.train(iters=1, bsize=1, rate=0.005, log_interval=2, schedule=12000)

print('Test: ', model.dnn_accuracy(model.test_data))
print('Train: ', model.dnn_accuracy(model.train_data))
print('Dev: ', model.dnn_accuracy(model.dev_data))

print('Total runtime: ', ((time.time() - start_time) / 3600.0))
