import random
import numpy as np
import matplotlib.pyplot as plt

from pysem.corpora import SNLI
from pysem.networks import RecurrentNetwork, DependencyNetwork
from pysem.utils.multiprocessing import flatten
from itertools import islice
from copy import deepcopy


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

        if isinstance(self.encoder, DependencyNetwork):
            self.acc.append(self.dnn_accuracy(self.dev_data))
            self._train_recursive_model()

    def _log_status(self, n):
        '''Keep track of training progress to log accuracy, print status.'''
        if n % self.schedule == 0 and n != 0:
            self.rate = self.rate / 2.0
            print('Learning rate annealed to ', self.rate)

        if n % self.log_interval == 0 and n != 0:
            if isinstance(self.encoder, RecurrentNetwork):
                self.acc.append(self.rnn_accuracy(self.dev_data))
            if isinstance(self.encoder, DependencyNetwork):
                self.acc.append(self.dnn_accuracy(self.dev_data))

    def _train_recurrent_model(self):
        '''Adapt training regime to accomodate recurrent encoder structure.'''
        for n in range(self.iters):
            self.encoder_copy = deepcopy(self.encoder)
            batch = random.sample(self.train_data, self.bsize)
            s1s = [sample.sentence1 for sample in batch]
            s2s = [sample.sentence2 for sample in batch]
            ys = SNLI.binarize([sample.label for sample in batch])

            self.training_iteration(s1s, s2s, ys)
            self.rnn_encoder_update()
            self._log_status(n)

        self.acc.append(self.rnn_accuracy(self.dev_data))

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

    def rnn_encoder_update(self):
        '''Update an RNN encoder based on the current training iteration by
        averaging the weights of the encoder and its copy.'''
        self.encoder.whh = (self.encoder.whh + self.encoder_copy.whh) / 2.
        self.encoder.why = (self.encoder.why + self.encoder_copy.why) / 2.
        self.encoder.bh = (self.encoder.bh + self.encoder_copy.bh) / 2.
        self.encoder.by = (self.encoder.by + self.encoder_copy.by) / 2.

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


def bow_accuracy(data, classifier, embedding_matrix, vectorizer):
    data = (x for x in data)  # convert to generator to use islice
    n_correct = 0
    n_total = 0
    batchsize = 5000

    while True:
        batch = list(islice(data, batchsize))
        n_total += len(batch)
        if len(batch) == 0:
            break

        s1s = [sample.sentence1 for sample in batch]
        s2s = [sample.sentence2 for sample in batch]

        s1_indicators = vectorizer.transform(s1s).toarray().T
        s2_indicators = vectorizer.transform(s2s).toarray().T

        s1_embeddings = np.dot(embedding_matrix, s1_indicators)
        s2_embeddings = np.dot(embedding_matrix, s2_indicators)

        xs = np.vstack((s1_embeddings, s2_embeddings))
        ys = SNLI.binarize([sample.label for sample in batch])

        predictions = classifier.predict(xs)
        n_correct += sum(np.equal(predictions, np.argmax(ys, axis=0)))

    return n_correct / n_total
