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

model = CompositeModel(snli, encoder, classifier)

start_time = time.time()

model.train(iters=1, bsize=1, rate=0.005, log_interval=2, schedule=12000)

