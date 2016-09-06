import random
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from pysem.corpora import SNLI
from pysem.utils.multiprocessing import flatten
from itertools import islice


class ExperimentManager(object):

    def __init__(self, data, sen_encoder, classifier):
        self.sen_encoder = sen_encoder
        self.classifier = classifier
        self.acc = []

        self.train_data = [d for d in data.train_data if d.label != '-']
        self.dev_data = [d for d in data.dev_data if d.label != '-']
        self.test_data = [d for d in data.test_data if d.label != '-']

    def train(self, epochs, batchsize, rate=0.01):
        iters = round((len(self.train_data) * epochs) / batchsize)

        for _ in range(iters):
            if _ % 200 == 0 and _ != 0:
                print('On iteration ', _)

            if _ % 2000 == 0 and _ != 0:
                self.acc.append(self.rnn_accuracy())

            batch = random.sample(self.train_data, batchsize)

            s1s = [sample.sentence1 for sample in batch]
            s2s = [sample.sentence2 for sample in batch]

            s1_encoder = self.sen_encoder
            s2_encoder = deepcopy(self.sen_encoder)

            s1_encoder.forward_pass(s1s)
            s2_encoder.forward_pass(s2s)

            s1_embs = s1_encoder.get_root_embedding()
            s2_embs = s2_encoder.get_root_embedding()

            xs = np.concatenate((s1_embs, s2_embs))
            ys = SNLI.binarize([sample.label for sample in batch])

            self.classifier.train(xs, ys, rate=rate)

            emb1_grad = self.classifier.yi_grad[:s1_encoder.dim]
            emb2_grad = self.classifier.yi_grad[s1_encoder.dim:]

            s1_encoder.backward_pass(emb1_grad, rate=rate)
            s2_encoder.backward_pass(emb2_grad, rate=rate)

            # update encoder weights by averaging across sentences
            self.sen_encoder.whh = (s1_encoder.whh + s2_encoder.whh) / 2
            self.sen_encoder.why = (s1_encoder.why + s2_encoder.why) / 2
            self.sen_encoder.bh = (s1_encoder.bh + s2_encoder.bh) / 2
            self.sen_encoder.by = (s1_encoder.by + s2_encoder.by) / 2

            word_set = set(flatten(s1_encoder.batch + s2_encoder.batch))
            word_set = [w.lower() for w in word_set if w != 'PAD']

            for word in word_set:
                if word in self.sen_encoder.vocab:
                    s1_vec = np.copy(s1_encoder.vectors[word])
                    s2_vec = np.copy(s2_encoder.vectors[word])
                    self.sen_encoder.vectors[word] = (s1_vec + s2_vec) / 2

    def rnn_accuracy(self):
        data = (x for x in self.dev_data)
        n_correct = 0
        n_total = 0
        batchsize = 100

        while True:
            batch = list(islice(data, batchsize))
            n_total += len(batch)
            if len(batch) == 0:
                break

            s1s = [sample.sentence1 for sample in batch]
            s2s = [sample.sentence2 for sample in batch]

            s1_encoder = self.sen_encoder
            s2_encoder = deepcopy(self.sen_encoder)

            s1_encoder.forward_pass(s1s)
            s2_encoder.forward_pass(s2s)

            s1_embs = s1_encoder.get_root_embedding()
            s2_embs = s2_encoder.get_root_embedding()

            xs = np.concatenate((s1_embs, s2_embs))
            ys = SNLI.binarize([sample.label for sample in batch])

            predictions = self.classifier.predict(xs)
            n_correct += sum(np.equal(predictions, np.argmax(ys, axis=0)))

        return n_correct / n_total


class DependencyNetworkManager(object):

    def __init__(self, data, sen_encoder, classifier):
        self.sen_encoder = sen_encoder
        self.classifier = classifier
        self.acc = []

        self.train_data = [d for d in data.train_data if d.label != '-']
        self.dev_data = [d for d in data.dev_data if d.label != '-']
        self.test_data = [d for d in data.test_data if d.label != '-']

    def train(self, epochs, batchsize=1, rate=0.01):
        iters = round((len(self.train_data) * epochs) / batchsize)

        for _ in range(iters):
            if _ % 200 == 0 and _ != 0:
                print('On iteration ', _)

            if _ % 2000 == 0 and _ != 0:
                self.acc.append(self.dnn_accuracy())

            batch = random.sample(self.train_data, batchsize)
            for sample in batch:

                s1_encoder = self.sen_encoder
                s2_encoder = deepcopy(self.sen_encoder)

                s1_encoder.forward_pass(sample.sentence1)
                s2_encoder.forward_pass(sample.sentence2)

                s1_emb = s1_encoder.get_root_embedding()
                s2_emb = s2_encoder.get_root_embedding()

                xs = np.concatenate((s1_emb, s2_emb))
                ys = SNLI.binarize([sample.label for sample in batch])

                self.classifier.train(xs, ys, rate=rate)

                emb1_grad = self.classifier.yi_grad[:s1_encoder.dim]
                emb2_grad = self.classifier.yi_grad[s1_encoder.dim:]

                w1_set = [node.lower_ for node in s1_encoder.tree]
                w2_set = [node.lower_ for node in s2_encoder.tree]

                s1_encoder.backward_pass(emb1_grad, rate=rate)
                s2_encoder.backward_pass(emb2_grad, rate=rate)

                s1_encoder.tree = None
                s2_encoder.tree = None

                # update encoder weights by averaging across sentences
                for dep in self.sen_encoder.deps:
                    w1 = np.copy(s1_encoder.weights[dep])
                    w2 = np.copy(s2_encoder.weights[dep])
                    self.sen_encoder.weights[dep] = (w1 + w2) / 2

                    b1 = np.copy(s1_encoder.biases[dep])
                    b2 = np.copy(s2_encoder.biases[dep])
                    self.sen_encoder.biases[dep] = (b1 + b2) / 2

                word_set = set(w1_set + w2_set)

                for word in word_set:
                    if word in self.sen_encoder.vocab:
                        s1_vec = np.copy(s1_encoder.vectors[word])
                        s2_vec = np.copy(s2_encoder.vectors[word])
                        self.sen_encoder.vectors[word] = (s1_vec + s2_vec) / 2

    def dnn_accuracy(self):
        data = (x for x in self.dev_data)
        count = 0
        label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        for sample in data:
            label = sample.label
            self.s1_encoder.forward_pass(sample.sentence1)
            self.s2_encoder.forward_pass(sample.sentence2)
            s1 = self.s1_encoder.get_root_embedding()
            s2 = self.s2_encoder.get_root_embedding()

            xs = np.concatenate((s1, s2))
            prediction = self.classifier.predict(xs)

            pred = label_dict[prediction[0]]
            if pred == label:
                count += 1

        return count / len(data)


def dnn_accuracy(data, classifier, s1_dnn, s2_dnn):
    count = 0
    label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    for sample in data:
        label = sample.label
        s1_dnn.forward_pass(sample.sentence1)
        s2_dnn.forward_pass(sample.sentence2)
        s1 = s1_dnn.get_root_embedding()
        s2 = s2_dnn.get_root_embedding()

        xs = np.concatenate((s1, s2))
        prediction = classifier.predict(xs)

        pred = label_dict[prediction[0]]
        if pred == label:
            count += 1

    return count / len(data)


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


def rnn_accuracy(data, classifier, s1_rnn, s2_rnn):
    data = (x for x in data)  # convert to generator to use islice
    n_correct = 0
    n_total = 0
    batchsize = 100

    while True:
        batch = list(islice(data, batchsize))
        n_total += len(batch)
        if len(batch) == 0:
            break

        s1s = [sample.sentence1 for sample in batch]
        s2s = [sample.sentence2 for sample in batch]

        s1_rnn.forward_pass(s1s)
        s2_rnn.forward_pass(s2s)

        s1 = s1_rnn.get_root_embedding()
        s2 = s2_rnn.get_root_embedding()

        xs = np.concatenate((s1, s2))
        ys = SNLI.binarize([sample.label for sample in batch])

        predictions = classifier.predict(xs)
        n_correct += sum(np.equal(predictions, np.argmax(ys, axis=0)))

    return n_correct / n_total


def plot(classifier, acc, acc_interval, scale=1):
    interval = acc_interval * scale

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(range(len(classifier.costs)), classifier.costs, 'g-')
    ax2.plot(range(0, len(classifier.costs) + 1, interval), acc, 'b-')

    ax1.set_xlabel('Minibatches')
    ax1.set_ylabel('Cost', color='g')
    ax2.set_ylabel('Dev Set Accuracy', color='b')

    plt.show()
