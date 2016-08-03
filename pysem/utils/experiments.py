import numpy as np
import matplotlib.pyplot as plt

from pysem.corpora import SNLI
from itertools import islice


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


def dnn_accuracy(data, classifier, s1_dnn, s2_dnn):
    count = 0
    label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    for sample in data:
        s1 = sample.sentence1
        s2 = sample.sentence2
        label = sample.label

        s1_dnn.forward_pass(s1)
        s2_dnn.forward_pass(s2)

        s1 = s1_dnn.get_root_embedding()
        s2 = s2_dnn.get_root_embedding()

        xs = np.concatenate((s1, s2))
        prediction = classifier.predict(xs)

        pred = label_dict[prediction[0]]
        if pred == label:
            count += 1

    return count / len(data)


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
