import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork, DependencyNetwork, FlatZeros
from pysem.utils.multiprocessing import flatten
from itertools import islice
from collections import defaultdict
from copy import deepcopy

from sklearn.feature_extraction.text import CountVectorizer


class CompositeModel(object):
    '''Combines a sentence encoder with a classifier to create a complete
    model for predicting inferential relationships on the SNLI dataset.
    '''
    def __init__(self, data, encoder, classifier):
        data.load_xy_pairs()

        self.encoder = encoder
        self.classifier = classifier
        self.acc = []

        self.train_data = [d for d in data.train_data if d.label != '-']
        self.dev_data = [d for d in data.dev_data if d.label != '-']
        self.test_data = [d for d in data.test_data if d.label != '-']

    @staticmethod
    def average(array, acc, span):
        for _ in range(round(len(array) / span) + 1):
            denom = len(array[:span])
            if denom > 0:
                val = sum(array[:span]) / span
                acc.append(val)
                array = array[span:]
        return acc

    def train(self, iters, bsize, rate=0.01, log_interval=1000, schedule=1000):
        '''Train the model for the specified number of iterations.'''
        self.iters = iters
        self.bsize = bsize
        self.rate = rate
        self.log_interval = log_interval
        self.schedule = schedule

        if isinstance(self.encoder, RecurrentNetwork):
            self.acc.append(self.rnn_accuracy(self.dev_data))
            self._train_recurrent_model()

        elif isinstance(self.encoder, DependencyNetwork):
            self.acc.append(self.dnn_accuracy(self.dev_data))
            self._train_recursive_model()

        elif isinstance(self.encoder, BagOfWords):
            self.acc.append(self.rnn_accuracy(self.dev_data))
            self._train_bow_model()

        elif isinstance(self.encoder, ProductOfWords):
            self.acc.append(self.dnn_accuracy(self.dev_data))
            self._train_product_model()

    def _log_status(self, n):
        '''Keep track of training progress to log accuracy, print status.'''
        if n % self.schedule == 0 and n != 0:
            self.rate = self.rate / 2.0
            print('Learning rate annealed to ', self.rate)

        if n % self.log_interval == 0 and n != 0:
            if isinstance(self.encoder, RecurrentNetwork):
                self.acc.append(self.rnn_accuracy(self.dev_data))
            elif isinstance(self.encoder, DependencyNetwork):
                self.acc.append(self.dnn_accuracy(self.dev_data))
            elif isinstance(self.encoder, BagOfWords):
                self.acc.append(self.rnn_accuracy(self.dev_data))
            elif isinstance(self.encoder, ProductOfWords):
                self.acc.append(self.dnn_accuracy(self.dev_data))

    def _train_bow_model(self):
        for n in range(self.iters):
            self.encoder_copy = deepcopy(self.encoder)
            batch = random.sample(self.train_data, self.bsize)
            s1s = [sample.sentence1 for sample in batch]
            s2s = [sample.sentence2 for sample in batch]
            ys = SNLI.binarize([sample.label for sample in batch])

            self.training_iteration(s1s, s2s, ys)
            self.bow_encoder_update()
            self._log_status(n)

        self.acc.append(self.rnn_accuracy(self.dev_data))

    def _train_recurrent_model(self):
        '''Adapt training regime to accomodate recurrent encoder structure.'''
        for n in range(self.iters):

            if n % 1000 == 0 and n != 0:
                print('Completed ', n, ' training iterations.')

            self.encoder_copy = deepcopy(self.encoder)
            batch = random.sample(self.train_data, self.bsize)
            s1s = [sample.sentence1 for sample in batch]
            s2s = [sample.sentence2 for sample in batch]
            ys = SNLI.binarize([sample.label for sample in batch])

            self.training_iteration(s1s, s2s, ys)
            self.rnn_encoder_update()
            self._log_status(n)

        self.acc.append(self.rnn_accuracy(self.dev_data))

    def _train_product_model(self):
        '''Adapt training regime to accomodate recursive encoder structure.'''
        for n in range(self.iters):
            self.encoder.tree = None
            self.encoder_copy = deepcopy(self.encoder)
            batch = random.sample(self.train_data, self.bsize)
            w1 = []
            w2 = []
            for sample in batch:
                s1 = sample.sentence1
                s2 = sample.sentence2
                ys = SNLI.binarize([sample.label])
                self.training_iteration(s1, s2, ys)

                w1 += [w for w in self.encoder.words]
                w2 += [w for w in self.encoder_copy.words]

            self.word_set = set(w1 + w2)
            self.prod_encoder_update()
            self._log_status(n)

        self.acc.append(self.dnn_accuracy(self.dev_data))

    def _train_recursive_model(self):
        '''Adapt training regime to accomodate recursive encoder structure.'''
        for n in range(self.iters):
            self.encoder.tree = None
            self.encoder_copy = deepcopy(self.encoder)
            batch = random.sample(self.train_data, self.bsize)
            w1 = []
            w2 = []
            for sample in batch:
                s1 = sample.sentence1
                s2 = sample.sentence2
                ys = SNLI.binarize([sample.label])
                self.training_iteration(s1, s2, ys)

                w1 += [n.lower_ for n in self.encoder.tree]
                w2 += [n.lower_ for n in self.encoder_copy.tree]

            self.word_set = set(w1 + w2)
            self.dnn_encoder_update()
            self._log_status(n)

        self.acc.append(self.dnn_accuracy(self.dev_data))

    def training_iteration(self, s1, s2, ys):
        '''Use input sentences to compute weight updates on encoder.'''
        self.encoder.forward_pass(s1)
        self.encoder_copy.forward_pass(s2)

        s1_emb = self.encoder.get_root_embedding()
        s2_emb = self.encoder_copy.get_root_embedding()

        xs = np.concatenate((s1_emb, s2_emb))
        self.classifier.train(xs, ys, rate=self.rate)

        emb1_grad = self.classifier.yi_grad[:self.encoder.dim]
        emb2_grad = self.classifier.yi_grad[self.encoder.dim:]

        self.encoder.backward_pass(emb1_grad, rate=self.rate)
        self.encoder_copy.backward_pass(emb2_grad, rate=self.rate)

    def bow_encoder_update(self):
        avg = (self.encoder.matrix + self.encoder_copy.matrix) / 2.0
        self.encoder.matrix = avg

    def rnn_encoder_update(self):
        '''Update an RNN encoder based on the current training iteration by
        averaging the weights of the encoder and its copy.'''
        for p in self.encoder.params:
            avg = (self.encoder.params[p] + self.encoder_copy.params[p]) / 2.0
            self.encoder.params[p] = avg

        word_set = set(flatten(self.encoder.batch + self.encoder_copy.batch))
        word_set = [w.lower() for w in word_set if w != 'PAD']

        for word in word_set:
            if word in self.encoder.vocab:
                s1_vec = np.copy(self.encoder.vectors[word])
                s2_vec = np.copy(self.encoder_copy.vectors[word])
                self.encoder.vectors[word] = (s1_vec + s2_vec) / 2.

    def dnn_encoder_update(self):
        '''Update a DNN encoder based on the current training iteration by
        averaging the weights of the encoder and its copy.'''
        for dep in self.encoder.deps:
            w1 = np.copy(self.encoder.weights[dep])
            w2 = np.copy(self.encoder_copy.weights[dep])
            self.encoder.weights[dep] = (w1 + w2) / 2.

            b1 = np.copy(self.encoder.biases[dep])
            b2 = np.copy(self.encoder_copy.biases[dep])
            self.encoder.biases[dep] = (b1 + b2) / 2.

        for word in self.word_set:
            if word in self.encoder.vocab:
                s1_vec = np.copy(self.encoder.vectors[word])
                s2_vec = np.copy(self.encoder_copy.vectors[word])
                self.encoder.vectors[word] = (s1_vec + s2_vec) / 2.

    def prod_encoder_update(self):
        '''Update a Product encoder based on the current training iteration by
        averaging the weights of the encoder and its copy.'''
        for word in self.word_set:
            if word in self.encoder.vocab:
                s1_vec = np.copy(self.encoder.vectors[word])
                s2_vec = np.copy(self.encoder_copy.vectors[word])
                self.encoder.vectors[word] = (s1_vec + s2_vec) / 2.

    def predict(self, s1, s2):
        label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        if isinstance(self.encoder, DependencyNetwork):
            self.encoder.forward_pass(s1)
            s1_emb = self.encoder.get_root_embedding()
            self.encoder.forward_pass(s2)
            s2_emb = self.encoder.get_root_embedding()

        else:
            self.encoder.forward_pass([s1])
            s1_emb = self.encoder.get_root_embedding()
            self.encoder.forward_pass([s2])
            s2_emb = self.encoder.get_root_embedding()

        xs = np.concatenate((s1_emb, s2_emb))
        prediction = self.classifier.predict(xs)

        return label_dict[prediction[0]]

    def dnn_accuracy(self, data):
        count = 0
        label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        for sample in data:
            label = sample.label

            self.encoder.forward_pass(sample.sentence1)
            s1_emb = self.encoder.get_root_embedding()

            self.encoder.forward_pass(sample.sentence2)
            s2_emb = self.encoder.get_root_embedding()

            xs = np.concatenate((s1_emb, s2_emb))
            prediction = self.classifier.predict(xs)

            pred = label_dict[prediction[0]]
            if pred == label:
                count += 1

        return count / len(data)

    def rnn_accuracy(self, data):
        data = (x for x in data)
        n_correct = 0
        n_total = 0

        while True:
            batch = list(islice(data, 500))
            n_total += len(batch)
            if len(batch) == 0:
                break

            s1s = [sample.sentence1 for sample in batch]
            s2s = [sample.sentence2 for sample in batch]

            s1_encoder = self.encoder
            s2_encoder = deepcopy(self.encoder)

            s1_encoder.forward_pass(s1s)
            s2_encoder.forward_pass(s2s)

            s1_embs = s1_encoder.get_root_embedding()
            s2_embs = s2_encoder.get_root_embedding()

            xs = np.concatenate((s1_embs, s2_embs))
            ys = SNLI.binarize([sample.label for sample in batch])

            predictions = self.classifier.predict(xs)
            n_correct += sum(np.equal(predictions, np.argmax(ys, axis=0)))

        return n_correct / n_total

    def plot(self, noshow=False):
        if isinstance(self.encoder, DependencyNetwork):
            avg_costs = self.average(self.classifier.costs, [], self.bsize)
            self.classifier.costs = avg_costs
        elif isinstance(self.encoder, ProductOfWords):
            avg_costs = self.average(self.classifier.costs, [], self.bsize)
            self.classifier.costs = avg_costs

        intr = self.log_interval
        xlen = len(self.classifier.costs)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.plot(range(xlen), self.classifier.costs, 'g-')
        ax2.plot(range(0, xlen + 1, intr), self.acc, 'b-')

        ax1.set_xlabel('Minibatches')
        ax1.set_ylabel('Cost', color='g')
        ax2.set_ylabel('Dev Set Accuracy', color='b')
        if not noshow:
            plt.show()


class BagOfWords(object):

    def __init__(self, dim, vocab, pretrained=False):
        self.dim = dim
        self.vectorizer = CountVectorizer(binary=True)
        self.vectorizer.fit(vocab)
        self.vocab = self.vectorizer.get_feature_names()
        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

    def forward_pass(self, batch):
        self.bsize = len(batch)
        self.indicators = self.vectorizer.transform(batch).toarray().T
        self.embeddings = np.dot(self.matrix, self.indicators)

    def backward_pass(self, error_grad, rate=0.1):
        matrix_grad = np.dot(error_grad, self.indicators.T) / self.bsize
        self.matrix -= rate * matrix_grad

    def get_root_embedding(self):
        return self.embeddings

    def pretrained_vecs(self, path):
        dim = self.dim
        idx_lookup = {word: idx for idx, word in enumerate(self.vocab)}
        self.matrix = np.zeros((dim, len(self.vocab)))

        with open(path, 'rb') as pfile:
            word2vec = pickle.load(pfile)

        for word in self.vocab:
            idx = idx_lookup[word]
            try:
                embedding = word2vec[word]
            except KeyError:
                scale = 1 / np.sqrt(dim)
                embedding = np.random.normal(loc=0, scale=scale, size=dim)

            self.matrix[:, idx] = embedding

    def random_vecs(self):
        scale = 1 / np.sqrt(self.dim)
        size = (self.dim, len(self.vectorizer.get_feature_names()))
        self.matrix = np.random.normal(loc=0, scale=scale, size=size)


class ProductOfWords(object):

    def __init__(self, dim, vocab, pretrained=False):
        self.dim = dim
        self.vocab = vocab
        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

    def forward_pass(self, sentence):
        self.embedding = np.ones((self.dim, 1))
        self.words = [t.text for t in DependencyNetwork.parser(sentence)]

        for word in self.words:
            try:
                self.embedding *= self.vectors[word]
            except KeyError:
                pass

    def backward_pass(self, error_grad, rate=0.1):
        self.wgrads = defaultdict(FlatZeros(self.dim))
        self.rate = rate

        for word in set(self.words):
            try:
                multiplier = self.embedding / self.vectors[word]
                self.wgrads[word] = error_grad * multiplier
            except KeyError:
                pass

        for word in self.wgrads:
            try:
                count = sum([1 for x in self.words if x == word])
                self.vectors[word] -= self.rate * self.wgrads[word] / count
            except KeyError:
                pass

    def get_root_embedding(self):
        return self.embedding

    def pretrained_vecs(self, path):
        '''Load pretrained word embeddings for initialization.'''
        self.vectors = {}
        with open(path, 'rb') as pfile:
            pretrained = pickle.load(pfile)

        for word in self.vocab:
            try:
                self.vectors[word] = pretrained[word].reshape(self.dim, 1)
            except KeyError:
                scale = 1 / np.sqrt(self.dim)
                randvec = np.random.normal(0, scale=scale, size=(self.dim, 1))
                self.vectors[word] = randvec

    def random_vecs(self):
        '''Use random word embeddings for initialization.'''
        scale = 1 / np.sqrt(self.dim)
        self.vectors = {word: np.random.normal(loc=0, scale=scale,
                        size=(self.dim, 1)) for word in self.vocab}
