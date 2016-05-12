class Model(object):
    """ 
    """
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


class MLP(Model):
    """
    A three layer neural network for performing classification. 
    
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
        self.w1 = np.random.random((dh, di+1))*eps*2-eps
        self.w2 = np.random.random((do, dh+1))*eps*2-eps
        self.costs = []

    def get_activations(self, x):
        self.yh = self.sigmoid(np.dot(self.w1, x.T))
        self.yh = np.append(np.ones((1, len(x))), self.yh, axis=0)
        self.yo = self.softmax(np.dot(self.w2, self.yh))

    def train(self, data, targs, iters, bsize=200, rate=0.15):       
        self.bsize = bsize
        for _ in range(iters):
            indices = np.random.randint(0,len(data), self.bsize)       
            x = data[indices, :]
            self.t = self.binarize(targs[indices])
            
            # Compute activations
            self.get_activations(x)
            
            # Compute gradients               
            yo_grad = self.yo-self.t
            yh_grad = np.dot(self.w2.T, yo_grad)*(self.yh*(1-self.yh))
            
            w1_grad = np.dot(yh_grad[1:,:], x) / self.bsize
            w2_grad = np.dot(yo_grad, self.yh.T) / self.bsize
            
            # Update weights
            self.w1 += -rate * w1_grad
            self.w2 += -rate * w2_grad
                           
            # Log the cost of the current weights
            self.costs.append(self.get_cost())
            
    def get_cost(self):
        return np.sum(-np.log(self.yo) * self.t) / float(self.bsize)
        
    def predict(self, data):
        self.get_activations(data)
        return np.argmax(self.yo, axis=0)

    def get_error(self, data, targs): 
        correct = sum(np.equal(self.predict(data), targs))
        return 100 * (1 - correct / float(len(targs)))       