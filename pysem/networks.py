import pickle
import spacy
import platform

import numpy as np

from collections import defaultdict
from handlers import SNLI
from spacy_utils import TokenWrapper

if platform.system() == 'Linux':
    snlipath = '/home/pblouw/corpora/snli_1.0/'
    wikipath = '/home/pblouw/corpora/wikipedia'
    cachepath = '/home/pblouw/cache/'
else:
    snlipath = '/Users/peterblouw/corpora/snli_1.0/'
    wikipath = '/Users/peterblouw/corpora/wikipedia'
    cachepath = '/Users/peterblouw/cache/'


class Model(object):
    """
    """
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_grad(x):
        return 1.0 - np.tanh(x)**2

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


def zeros(dim):
    def func():
        return np.zeros((dim, dim))
    return func


class MLP(Model):
    """
    A multilayer perceptron performing classification.
    Parameters:
    -----------
    di : int
        The dimensionality of the input vectors being classified.
    dh : int
        The dimensionality of the hidden layer of the network.
    do : int
        The dimensionality of the output vector that encodes a classification
        decision. (i.e. a probability distribution over labels)
    eps : float, optional
        Scaling factor on random weight initialization. By default, the
        weightsare chosen from a uniform distribution on the interval
        [-0.1, 0.1].
    """
    def __init__(self, di, dh, do, eps=0.1):
        temp = dh
        self.w1 = np.random.random((temp, di+1))*eps*2-eps
        self.w2 = np.random.random((do, dh+1))*eps*2-eps
        self.costs = []
        self.bsize = 1
        self.node_to_label = {0: 'entailment', 1: 'neutral', 2: 'contradicts'}

    def get_activations(self, x):
        self.yh = self.sigmoid(np.dot(self.w1, x))
        self.yh = np.vstack((np.ones(self.bsize), self.yh))

        self.yo = self.softmax(np.dot(self.w2, self.yh))

    def train(self, xs, ys, iters, bsize=1, rate=0.3):
        self.bsize = bsize
        xs = np.reshape(xs, (len(xs), 1))
        # BoW = CountVectorizer(binary=True)
        # BoW.fit(snli.vocab)

        # snli.extractor = snli.get_xy_pairs
        # data = {d for d in snli.train_data if d[1] != '-'}
        # print('Training set size: ', len(data))

        for _ in range(iters):

            # if _ % 10000 == 0:
            #     if _ != 0:
            #         print('Completed ', _, ' training iterations!')

            # if _ == 0.6 * iters:
            #     rate = 0.15
            #     print('Dropped rate to ', rate)
            # if _ == 0.8 * iters:
            #     rate = 0.075
            #     print('Dropped rate to ', rate)

            # batch = random.sample(data, self.bsize)

            # Turn batches into arrays
            # prems = [s[0][0] for s in batch]
            # hyps = [s[0][1] for s in batch]
            # targs = [s[1] for s in batch]

            # prem_bag = BoW.transform(prems).toarray().T
            # prem_bag = np.vstack((np.ones(self.bsize), prem_bag))

            # hyp_bag = BoW.transform(hyps).toarray().T

            # inp_bag = np.vstack((prem_bag, hyp_bag))
            self.targ_bag = self.binarize([ys])

            # Compute activations
            self.get_activations(xs)

            # Compute gradients
            yo_grad = self.yo-self.targ_bag
            yh_grad = np.dot(self.w2.T, yo_grad)*(self.yh*(1-self.yh))
            # w2_grad = np.dot(yo_grad, self.yh.T) / self.bsize
            # w1_grad = np.dot(yh_grad[1:, :], xs.T) / self.bsize

            self.yi_grad = np.dot(self.w1.T, yh_grad[1:])
            self.yo_grad = yo_grad
            self.yh_grad = yh_grad
            # Update weights
            # self.w1 += -rate * w1_grad
            # self.w2 += -rate * w2_grad

            # Log the cost of the current weights
            self.costs.append(self.get_cost())

    def get_cost(self):
        return np.sum(-np.log(self.yo) * self.targ_bag) / float(self.bsize)

    def predict(self, x):
        x = np.reshape(x, (len(x), 1))
        self.get_activations(x)
        return self.node_to_label[int(np.argmax(self.yo, axis=0))]

    def get_accuracy(self, snli, dataset):

        snli.extractor = snli.get_xy_pairs

        def tally(data):
            # Turn batches into arrays
            self.bsize = len(data)

            # prems = [s[0][0] for s in data]
            # hyps = [s[0][1] for s in data]
            # targs = [s[1] for s in data]

            # prem_bag = BoW.transform(prems).toarray().T
            # prem_bag = np.vstack((np.ones(len(data)), prem_bag))

            # hyp_bag = BoW.transform(hyps).toarray().T
            # inp_bag = np.vstack((prem_bag, hyp_bag))

            # targs = self.binarize(targs)

            # correct = sum(np.equal(self.predict(inp_bag),
            #               np.argmax(targs, axis=0)))

            return correct

        if dataset == 'dev':
            data = {d for d in snli.dev_data if d[1] != '-'}
            self.bsize = len(data)
            return tally(data) / float(self.bsize)

        if dataset == 'train':
            correct = 0
            bsize = 10000
            for _ in range(55):
                data = []
                for __ in range(bsize):
                    try:
                        while True:
                            x = next(snli.train_data)
                            if x[1] != '-':
                                break
                        data.append(x)
                    except:
                        break
                correct += tally(data)

            snli.extractor = snli.get_xy_pairs
            return correct / float(len({d for d in snli.train_data
                                   if d[1] != '-'}))

    @staticmethod
    def binarize(label_list):
        lookup = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 1}
        y_idx = [lookup[l] for l in label_list]
        x_idx = range(len(label_list))

        vals = np.zeros((3, len(label_list)))
        vals[y_idx, x_idx] = 1
        return vals


class DependencyNetwork(Model):

    def __init__(self, embedding_dim, vocab, eps=0.1):
        self.dim = embedding_dim
        self.vocab = sorted(vocab)
        self.indices = {wrd: idx for idx, wrd in enumerate(self.vocab)}
        self.parser = spacy.load('en')
        self.vectors = np.random.random((len(vocab), self.dim))*eps*2-eps
        self.load_dependencies('dependencies.pickle')
        self.initialize_weights()
        self.wgrads = defaultdict(zeros(self.dim))

    @staticmethod
    def gaussian_id(dim):
        identity = np.eye(dim)
        gaussian = np.random.normal(loc=0, scale=0.05, size=(dim, dim))
        return identity + gaussian

    def load_dependencies(self, path):
        with open(path, 'rb') as pfile:
            self.depset = pickle.load(pfile)

    def initialize_weights(self):
        self.weights = {'token': self.gaussian_id(self.dim)}
        for dep in self.depset:
            self.weights[dep] = self.gaussian_id(self.dim)

    def reset_comp_graph(self):
        for token in self.tree:
            token.computed = False

    def reset_embeddings(self):
        for token in self.tree:
            token._embedding = None

    def compute_gradients(self):
        for token in self.tree:
            if not self.has_children(token):
                continue
            if token.gradient is not None:
                children = self.get_children(token)
                gradient = token.gradient
                for child in children:
                    if child.gradient is not None:
                        continue
                    self.wgrads[child.dep_] += np.outer(gradient,
                                                        child.embedding)
                    child.gradient = np.dot(self.weights[child.dep_].T,
                                            gradient)
                    nonlinearity = child.embedding * (1-child.embedding)
                    nonlinearity = nonlinearity.reshape((len(nonlinearity), 1))

                    child.gradient = child.gradient * nonlinearity
                    child.computed = True
                    print(child, ' Backpropped!')

        grads_computed = [t.computed for t in self.tree]
        if all(grads_computed):
            print('DONE')
            return
        else:
            self.compute_gradients()

    def forward_pass(self, sentence):
        self.tree = [TokenWrapper(t) for t in self.parser(sentence)]
        self.compute_leaves()
        self.compute_nodes()
        self.reset_comp_graph()

    def backward_pass(self, error_grad, rate=1.5):
        self.set_root_gradient(error_grad)
        self.compute_gradients()

        for dep in self.wgrads:
            self.weights[dep] += -rate * self.wgrads[dep]

        self.wgrads = defaultdict(zeros(self.dim))
        self.reset_comp_graph()
        self.reset_embeddings()

    def get_children(self, token):
        children = []
        for other_token in self.tree:
            if other_token.idx in [child.idx for child in token.children]:
                children.append(other_token)

        return children

    def has_children(self, token):
        if list(token.children) == list():
            return False
        else:
            return True

    def embed(self, token, children=list()):
        try:
            idx = self.indices[token.lower_]
            emb = np.dot(self.weights['token'], self.vectors[idx, :])
        except KeyError:
            emb = np.zeros(self.dim)

        for child in children:
            emb += np.dot(self.weights[child.dep_], child.embedding)

        token.embedding = self.tanh(emb)
        token.computed = True

    def compute_leaves(self):
        for token in self.tree:
            if not self.has_children(token):
                self.embed(token)

    def compute_nodes(self):
        for token in self.tree:
            if self.has_children(token) and not token.computed:
                children = self.get_children(token)
                children_computed = [child.computed for child in children]

                if all(children_computed):
                    self.embed(token, children)

        nodes_computed = [token.computed for token in self.tree]
        if all(nodes_computed):
            return
        else:
            self.compute_nodes()

    def set_root_gradient(self, grad):
        for token in self.tree:
            if token.head.idx == token.idx:
                token.gradient = grad[1:]
                token.computed = True

    def get_sentence_embedding(self):
        for token in self.tree:
            if token.head.idx == token.idx:
                return token.embedding


snli = SNLI(snlipath)
snli.extractor = snli.get_xy_pairs
snli.load_vocab('snli_words')

# model = DependencyNetwork(4, snli.vocab)

# start_time = time.time()

# for _ in range(5000):
#     sentence = next(snli.train_data)[0]

#     model.forward_pass(sentence)

# print('Total runtime: ', time.time() - start_time)

# snli = SNLI(snlipath)
# snli.build_vocab()

data = next(snli.train_data)
# data = next(snli.train_data)
s1 = data[0][0]
label = data[1]

depnet = DependencyNetwork(embedding_dim=50, vocab=snli.vocab)
classifier = MLP(di=50, dh=50, do=3)


for _ in range(100):
    depnet.forward_pass(s1)

    bias = np.ones(1)
    svec = depnet.get_sentence_embedding()

    xs = np.concatenate((bias, svec))
    ys = label

    classifier.train(xs, ys, iters=1)

    depnet.backward_pass(classifier.yi_grad)

    print(classifier.predict(xs))
    print(label)
# print(classifier.yi_grad)
# print(classifier.yh_grad)
# print(classifier.yo_grad)

# model = MLP(2*len(BoW.get_feature_names()), 200, 3)
# print('Dev Set Accuracy Before: ', model.get_accuracy(snli, 'dev'))

# model.train(snli, iters=1000, bsize=100)

# print('Train Set Accuracy After: ', model.get_accuracy(snli, 'train'))
# print('Dev Set Accuracy After: ', model.get_accuracy(snli, 'dev'))

# plt.figure()
# plt.plot(np.arange(len(model.costs)), model.costs)
# plt.show()
