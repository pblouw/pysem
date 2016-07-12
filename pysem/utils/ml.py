import numpy as np


class Model(object):
    """Base class with activation functions and utilities shared by multiple
    machine learning models. Tanh is used throughout as a standard activation
    since it bounds vector elements between -1 and 1.
    """
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_grad(x):
        return 1.0 - x * x

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


class LogisticRegression(Model):
    """Performs multinomial logistic regression by predicting a probability
    distribution over class labels given a set of input features.

    Parameters:
    ----------
    n_features : int
        The number of features to be used for predicting class labels.
    n_labels : int
        The number of class labels to be predicted from input features.
    eps: float, optional
        The size of the interval to sample from for weight initialization.
        Weights are sampled from a continuous uniform distribution over the
        interval [-eps, eps].

    Attributes:
    -----------
    weights : numpy.ndarray
        The weight matrix used to compute the predicted probability of each
        label.
    bias : numpy.ndarray
        The bias on the output layer that computes probabilities over labels.
    costs : list
        A list of costs associated with each batch that was trained on.
        Initialized to the empty list.
    """
    def __init__(self, n_features, n_labels, eps=0.1):
        self.weights = np.random.random((n_labels, n_features)) * eps
        self.bias = np.random.random(n_labels) * eps
        self.costs = []

    def train(self, xs, ys, rate):
        '''Train a regression model using gradient descent on the provided x-y
        pairs. The x and y data should be in the form of two numpy arrays with
        an equal number of columns, and paired data points should occupy
        corresponding columns in the two arrays.
        '''
        bsize = xs.shape[1]
        probs = self.get_probs(xs)

        self.o_grad = probs - ys
        self.w_grad = np.dot(self.o_grad, xs.T) / bsize
        self.b_grad = np.sum(self.o_grad, axis=1) / bsize
        self.yi_grad = np.dot(self.weights.T, self.o_grad)

        self.weights -= rate * self.w_grad
        self.bias -= rate * self.b_grad

        cost = np.sum(-np.log(probs) * ys) / bsize
        self.costs.append(cost)

    def predict(self, xs):
        '''Predict labels associated with a set of input feature vectors.'''
        probs = self.softmax(np.dot(self.weights, xs))
        return np.argmax(probs, axis=0)

    def get_cost(self, xs, ys):
        '''Evaluate cost of current parameters for gradient checking.'''
        probs = self.get_probs(xs)
        cost = np.sum(-np.log(probs) * ys) / xs.shape[1]
        return cost

    def get_probs(self, xs):
        '''Compute label probabilities for an array of feature vectors.'''
        bias = self.bias.reshape(len(self.bias), 1)  # for array broadcasting
        probs = self.softmax(np.dot(self.weights, xs) + bias)
        return probs


class MultiLayerPerceptron(Model):
    """
    A three layer MLP for performing classification on input vectors. This is
    vanilla neural network with a single hidden layer and no tricks for using
    fancy parameter updates or activation functions. Tanh is used by default.

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
        The size of the interval to sample from for weight initialization.
        Weights are sampled from a continuous uniform distribution over the
        interval [-eps, eps].

    Attributes:
    ----------
    w1 : numpy.ndarray
        The weights between the input layer and the hidden layer.
    w2 : numpy.ndarray
        The weights between the hidden layer and the output layer.
    bh : numpy.ndarray
        The bias vector for the hidden layer.
    bo : numpy.ndarray
        The bias vector for the output layer.
    costs : list
        A list of costs associated with each batch that was trained on.
        Initialized to the empty list.
    """
    def __init__(self, di, dh, do, eps=0.1):
        self.w1 = np.random.random((dh, di)) * eps * 2 - eps
        self.w2 = np.random.random((do, dh)) * eps * 2 - eps
        self.bh = np.random.random(dh) * eps * 2 - eps
        self.bo = np.random.random(do) * eps * 2 - eps
        self.costs = []
        self.tstep = 0

        # arrays for estimating first gradient moment for ADAM optimization
        self.w1_m = np.zeros_like(self.w1)
        self.w2_m = np.zeros_like(self.w2)
        self.bh_m = np.zeros_like(self.bh)
        self.bo_m = np.zeros_like(self.bo)

        # arrays for estimating second gradient moment for ADAM optimization
        self.w1_v = np.zeros_like(self.w1)
        self.w2_v = np.zeros_like(self.w2)
        self.bh_v = np.zeros_like(self.bh)
        self.bo_v = np.zeros_like(self.bo)

        self.b1 = 0.9
        self.b2 = 0.999
        self.epsilon = 1e-8

    def get_probs(self, xs):
        '''Compute label probabilities for an array of input vectors.'''
        bh = self.bh.reshape(len(self.bh), 1)
        bo = self.bo.reshape(len(self.bo), 1)

        self.yh = self.tanh(np.dot(self.w1, xs) + bh)
        self.yo = self.softmax(np.dot(self.w2, self.yh) + bo)

    def train(self, xs, ys, rate):
        '''Train a 3 layer MLP using gradient descent on the provided x-y
        pairs. The x and y data should be in the form of two numpy arrays with
        an equal number of columns, and paired data points should occupy
        corresponding columns in the two arrays.'''
        bsize = xs.shape[1]
        self.get_probs(xs)
        self.tstep += 1

        yo_grad = self.yo - ys
        yh_grad = np.dot(self.w2.T, yo_grad) * self.tanh_grad(self.yh)
        self.yi_grad = np.dot(self.w1.T, yh_grad)

        self.bo_grad = np.sum(yo_grad, axis=1) / bsize
        self.bh_grad = np.sum(yh_grad, axis=1) / bsize

        self.w2_grad = np.dot(yo_grad, self.yh.T) / bsize
        self.w1_grad = np.dot(yh_grad, xs.T) / bsize

        self.w1_m = self.b1 * self.w1_m + (1 - self.b1) * self.w1_grad
        self.w2_m = self.b1 * self.w2_m + (1 - self.b1) * self.w2_grad
        self.bh_m = self.b1 * self.bh_m + (1 - self.b1) * self.bh_grad
        self.bo_m = self.b1 * self.bo_m + (1 - self.b1) * self.bo_grad

        self.w1_v = self.b2 * self.w1_v + (1 - self.b2) * (self.w1_grad ** 2)
        self.w2_v = self.b2 * self.w2_v + (1 - self.b2) * (self.w2_grad ** 2)
        self.bh_v = self.b2 * self.bh_v + (1 - self.b2) * (self.bh_grad ** 2)
        self.bo_v = self.b2 * self.bo_v + (1 - self.b2) * (self.bo_grad ** 2)

        w1_m_est = self.w1_m / (1 - self.b1 ** self.tstep)
        w2_m_est = self.w2_m / (1 - self.b1 ** self.tstep)
        bo_m_est = self.bo_m / (1 - self.b1 ** self.tstep)
        bh_m_est = self.bh_m / (1 - self.b1 ** self.tstep)

        w1_v_est = self.w1_v / (1 - self.b2 ** self.tstep)
        w2_v_est = self.w2_v / (1 - self.b2 ** self.tstep)
        bo_v_est = self.bo_v / (1 - self.b2 ** self.tstep)
        bh_v_est = self.bh_v / (1 - self.b2 ** self.tstep)

        self.w1 -= rate * w1_m_est / (np.sqrt(w1_v_est) + self.epsilon)
        self.w2 -= rate * w2_m_est / (np.sqrt(w2_v_est) + self.epsilon)
        self.bh -= rate * bh_m_est / (np.sqrt(bh_v_est) + self.epsilon)
        self.bo -= rate * bo_m_est / (np.sqrt(bo_v_est) + self.epsilon)

        cost = np.sum(-np.log(self.yo) * ys) / bsize
        self.costs.append(cost)

    def predict(self, xs):
        '''Predict labels associated with a set of input feature vectors.'''
        self.get_probs(xs)
        return np.argmax(self.yo, axis=0)

    def get_cost(self, xs, ys):
        '''Evaluate cost of current parameters for gradient checking.'''
        self.get_probs(xs)
        cost = np.sum(-np.log(self.yo) * ys) / xs.shape[1]
        return cost
