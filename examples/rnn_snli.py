import time
import pickle

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork
from pysem.utils.ml import MultiLayerPerceptron
from pysem.utils.snli import CompositeModel

snli = SNLI('/Users/peterblouw/corpora/snli_1.0/')
snli.load_vocab('snli_words.pickle')

dim = 300
pretrained = 'pretrained_snli_embeddings.pickle'

encoder = RecurrentNetwork(dim=dim, vocab=snli.vocab, pretrained=pretrained)
classifier = MultiLayerPerceptron(di=2*dim, dh=dim, do=3)

model = CompositeModel(snli, encoder, classifier)

start_time = time.time()

model.train(iters=100000, bsize=100, rate=0.1, log_interval=10000, schedule=15000)
print('Test: ', model.rnn_accuracy(model.test_data))
print('Train: ', model.rnn_accuracy(model.train_data))
print('Dev: ', model.rnn_accuracy(model.dev_data))

print('Total runtime: ', ((time.time() - start_time) / 3600.0))

with open('rnn_model_w2v_high_learning_rate.pickle', 'wb') as pfile:
    pickle.dump(model, pfile)

