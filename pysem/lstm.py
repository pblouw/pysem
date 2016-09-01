import nltk
import numpy as np

from pysem.utils.spacy import TokenWrapper
from pysem.networks import RecursiveModel


class LSTM(RecursiveModel):
    """A recurrent network that uses LSTM cells in place of the usual hidden
    state representations.
    """
    def __init__(self, dim, vocab, pretrained=False):
        self.dim = dim
        self.vocab = sorted(vocab)

        # initialize input gate weights
        self.iW = self.random_weights(dim)
        self.iU = self.random_weights(dim)
        self.i_bias = 3 * np.ones(self.dim).reshape((self.dim, 1))

        # initialize forget gate weights
        self.fW = self.random_weights(dim)
        self.fU = self.random_weights(dim)
        self.f_bias = 3 * np.ones(self.dim).reshape((self.dim, 1))

        # initialize output gate weights
        self.oW = self.random_weights(dim)
        self.oU = self.random_weights(dim)
        self.o_bias = 3 * np.ones(self.dim).reshape((self.dim, 1))

        # initialize cell input weights
        self.uW = self.random_weights(dim)
        self.uU = self.random_weights(dim)
        self.u_bias = 0 * np.ones(self.dim).reshape((self.dim, 1))

        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

    def clip_grad(self, gradient, clipval=5):
        '''Clip a large gradient so that its norm is equal to clipval.'''
        norm = np.linalg.norm(gradient.flatten())
        if norm > clipval:
            gradient = (gradient / norm) * 5
        return gradient

    def compute_embeddings(self):
        '''Compute LSTM cell states for each item in the sequence.'''
        self.hs[-1] = np.zeros(self.dim).reshape((self.dim, 1))
        self.cell_states[-1] = np.zeros(self.dim).reshape((self.dim, 1))

        for i in range(len(self.sen)):
            vec = self.to_vector(self.sen[i])
            i_gate = np.dot(self.iW, vec) + np.dot(self.iU, self.hs[i-1])
            o_gate = np.dot(self.oW, vec) + np.dot(self.oU, self.hs[i-1])
            f_gate = np.dot(self.fW, vec) + np.dot(self.fU, self.hs[i-1])
            cell_input = np.dot(self.uW, vec) + np.dot(self.uU, self.hs[i-1])

            self.i_gates[i] = self.sigmoid(i_gate + self.i_bias)
            self.o_gates[i] = self.sigmoid(o_gate + self.o_bias)
            self.f_gates[i] = self.sigmoid(f_gate + self.f_bias)
            self.cell_inputs[i] = np.tanh(cell_input + self.u_bias)

            self.cell_states[i] = self.i_gates[i] * self.cell_inputs[i]
            self.cell_states[i] += self.f_gates[i] * self.cell_states[i-1]
            self.hs[i] = self.o_gates[i] * self.cell_states[i]
            self.xs[i] = vec

    def forward_pass(self, sen):
        '''Convert input sentence into sequence and compute cell states.'''
        self.sen = nltk.word_tokenize(sen)
        self.i_gates = {}
        self.f_gates = {}
        self.o_gates = {}
        self.cell_inputs = {}
        self.cell_states = {}
        self.hs = {}
        self.xs = {}
        self.compute_embeddings()

    def backward_pass(self, error_grad, rate=0.1):
        '''Compute gradients for all weights in the LSTM'''
        self.diW = np.zeros_like(self.iW)
        self.doW = np.zeros_like(self.oW)
        self.dfW = np.zeros_like(self.fW)
        self.duW = np.zeros_like(self.uW)

        self.diU = np.zeros_like(self.iU)
        self.doU = np.zeros_like(self.oU)
        self.dfU = np.zeros_like(self.fU)
        self.duU = np.zeros_like(self.uU)

        self.i_bias_grad = np.zeros_like(self.i_bias)
        self.f_bias_grad = np.zeros_like(self.f_bias)
        self.o_bias_grad = np.zeros_like(self.o_bias)
        self.u_bias_grad = np.zeros_like(self.u_bias)

        self.dxs = {w.lower(): np.zeros_like(self.i_bias) for w in self.sen}

        h_grad = error_grad
        s_grad = np.zeros(self.dim).reshape((self.dim, 1))

        # compute all gradients in reverse through the sequence
        for i in reversed(range(len(self.sen))):
            d_cell_state = h_grad * self.o_gates[i] + s_grad

            d_o_gate = h_grad * self.cell_states[i]
            d_o_gate *= self.sigmoid_grad(self.o_gates[i])

            d_f_gate = d_cell_state * self.cell_states[i-1]
            d_f_gate *= self.sigmoid_grad(self.f_gates[i])

            d_i_gate = d_cell_state * self.cell_inputs[i]
            d_i_gate *= self.sigmoid_grad(self.i_gates[i])

            d_cell_input = d_cell_state * self.i_gates[i]
            d_cell_input *= self.tanh_grad(self.cell_inputs[i])

            self.doW += np.dot(d_o_gate, self.xs[i].T)
            self.diW += np.dot(d_i_gate, self.xs[i].T)
            self.dfW += np.dot(d_f_gate, self.xs[i].T)
            self.duW += np.dot(d_cell_input, self.xs[i].T)

            self.doU += np.dot(d_o_gate, self.hs[i-1].T)
            self.diU += np.dot(d_i_gate, self.hs[i-1].T)
            self.dfU += np.dot(d_f_gate, self.hs[i-1].T)
            self.duU += np.dot(d_cell_input, self.hs[i-1].T)

            self.i_bias_grad += d_i_gate
            self.f_bias_grad += d_f_gate
            self.o_bias_grad += d_o_gate
            self.u_bias_grad += d_cell_input

            dx = np.zeros_like(self.xs[i])
            dh = np.zeros_like(self.hs[i-1])

            # gradients for input vecs and to pass to children
            dx += np.dot(self.oW.T, d_o_gate)
            dx += np.dot(self.iW.T, d_i_gate)
            dx += np.dot(self.fW.T, d_f_gate)
            dx += np.dot(self.uW.T, d_cell_input)
            self.dxs[self.sen[i].lower()] += dx

            dh += np.dot(self.oU.T, d_o_gate)
            dh += np.dot(self.iU.T, d_i_gate)
            dh += np.dot(self.fU.T, d_f_gate)
            dh += np.dot(self.uU.T, d_cell_input)

            # update gradients for use in next iteration
            h_grad = dh
            s_grad = self.f_gates[i] * d_cell_state

        # perform weight updates
        self.oW -= rate * self.doW
        self.fW -= rate * self.dfW
        self.iW -= rate * self.diW
        self.uW -= rate * self.duW

        self.oU -= rate * self.doU
        self.fU -= rate * self.dfU
        self.iU -= rate * self.diU
        self.uU -= rate * self.duU

        self.i_bias -= rate * self.i_bias_grad
        self.f_bias -= rate * self.f_bias_grad
        self.o_bias -= rate * self.o_bias_grad
        self.u_bias -= rate * self.u_bias_grad

        # update word embeddings
        for item in self.sen:
            word = item.lower()
            count = sum([1 for x in self.sen if word == x.lower()])
            try:
                self.vectors[word] -= rate * self.dxs[word] / count
            except KeyError:
                pass

    def to_vector(self, word):
        '''Get input vector for the word in a given sequence position.'''
        try:
            vector = np.copy(self.vectors[word.lower()])
        except KeyError:
            vector = np.zeros(self.dim).reshape((self.dim, 1))
        return vector

    def get_root_embedding(self):
        '''Returns the embeddings for the final/root node in the sequence.'''
        return np.copy(self.hs[len(self.sen)-1])


class TreeLSTM(RecursiveModel):
    """A dependency network that uses LSTM cells in place of the usual hidden
    state representations. This is currently a syntactically tied version of
    the network, in that the same weights are shared across all dependencies
    in the network's tree structure.
    """
    def __init__(self, dim, vocab, pretrained=False):
        self.dim = dim
        self.vocab = sorted(vocab)

        # initialize input gate weights
        self.iW = self.random_weights(dim)
        self.iU = self.random_weights(dim)
        self.i_bias = 5 * np.ones(self.dim).reshape((self.dim, 1))

        # initialize forget gate weights
        self.fW = self.random_weights(dim)
        self.fU = self.random_weights(dim)
        self.f_bias = 5 * np.ones(self.dim).reshape((self.dim, 1))

        # initialize output gate weights
        self.oW = self.random_weights(dim)
        self.oU = self.random_weights(dim)
        self.o_bias = 5 * np.ones(self.dim).reshape((self.dim, 1))

        # initialize cell input weights
        self.uW = self.random_weights(dim)
        self.uU = self.random_weights(dim)
        self.u_bias = 5 * np.ones(self.dim).reshape((self.dim, 1))

        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

    def reset_comp_graph(self):
        '''Flag all nodes in the graph as being uncomputed.'''
        for node in self.tree:
            node.computed = False

    def clip_gradient(self, gradient, clipval=5):
        '''Clip a large gradient so that its norm is equal to clipval.'''
        norm = np.linalg.norm(gradient)
        if norm > clipval:
            return (gradient / norm) * clipval
        else:
            return gradient

    def compute_embeddings(self):
        '''Computes embeddings for all nodes in the graph by recursively
        computing the embeddings for nodes whose children have all been
        computed. Recursion terminates when every node has an embedding.'''
        for node in self.tree:
            if not node.computed:
                children = self.get_children(node)
                children_computed = [c.computed for c in children]

                if all(children_computed):
                    self.update_node(node, children)

        nodes_computed = [node.computed for node in self.tree]
        if all(nodes_computed):
            return
        else:
            self.compute_embeddings()

    def compute_gradients(self):
        '''
        '''
        for node in self.tree:
            parent = self.get_parent(node)

            if not node.computed and parent.computed:
                node.top_grad = self.clip_gradient(node.top_grad)
                c_grad = node.o_gate * node.top_grad
                c_grad += parent.c_grad * parent.f_gates[node.idx]
                node.c_grad = c_grad

                self.grad_update(node, parent)

        if all([node.computed for node in self.tree]):
            return
        else:
            self.compute_gradients()

    def grad_update(self, node, parent):

        inp_grad = np.zeros_like(node.c_grad)
        emb_grad = np.zeros_like(node.c_grad)

        # gradients for output gate
        o_grad = node.cell_state * node.top_grad
        node.o_grad = o_grad * self.sigmoid_grad(node.o_gate)

        self.doW += np.dot(node.o_grad, node.inp_vec.T)
        self.doU += np.dot(node.o_grad, node.h_tilda.T)

        # gradients for input gate
        i_grad = node.c_grad * node.cell_input
        node.i_grad = i_grad * self.sigmoid_grad(node.i_gate)

        self.diW += np.dot(node.i_grad, node.inp_vec.T)
        self.diU += np.dot(node.i_grad, node.h_tilda.T)

        # gradients for cell input
        ci_grad = node.i_gate * node.c_grad
        node.ci_grad = ci_grad * self.tanh_grad(node.cell_input)

        self.duW += np.dot(node.ci_grad, node.inp_vec.T)
        self.duU += np.dot(node.ci_grad, node.h_tilda.T)

        # gradients for biases
        self.i_bias_grad += node.i_grad
        self.o_bias_grad += node.o_grad
        self.u_bias_grad += node.ci_grad

        # gradients for input vecs and to pass to children
        inp_grad += np.dot(self.diW.T, node.i_grad)
        inp_grad += np.dot(self.doW.T, node.o_grad)
        inp_grad += np.dot(self.duW.T, node.ci_grad)

        emb_grad += np.dot(self.diU.T, node.i_grad)
        emb_grad += np.dot(self.doU.T, node.o_grad)
        emb_grad += np.dot(self.duU.T, node.ci_grad)

        # gradients for forget gates
        node.f_grads = {}
        for child in self.get_children(node):
            f_grad = node.c_grad * child.cell_state
            f_grad = f_grad * self.sigmoid_grad(node.f_gates[child.idx])
            node.f_grads[child.idx] = f_grad
            self.f_bias_grad += f_grad

            inp_grad += np.dot(self.dfW.T, f_grad)
            child.top_grad = emb_grad
            child.top_grad += np.dot(self.dfU.T, f_grad)

            self.dfW += np.dot(f_grad, node.inp_vec.T)
            self.dfU += np.dot(f_grad, child.embedding.T)

        node.inp_grad = inp_grad
        node.computed = True

    def update_node(self, node, children):
        '''Compute the state of the LSTM cell corresponding to the supplied
        node in the parse tree, given the nodes corresponding to its children.
        '''
        if len(children) > 0:
            node.h_tilda = sum([node.embedding for node in children])
        else:
            node.h_tilda = np.zeros(self.dim).reshape((self.dim, 1))
        try:
            node.inp_vec = self.vectors[node.lower_]
        except KeyError:
            node.inp_vec = np.zeros(self.dim).reshape((self.dim, 1))

        i_gate = np.dot(self.iW, node.inp_vec) + np.dot(self.iU, node.h_tilda)
        node.i_gate = self.sigmoid(i_gate + self.i_bias)

        o_gate = np.dot(self.oW, node.inp_vec) + np.dot(self.oU, node.h_tilda)
        node.o_gate = self.sigmoid(o_gate + self.o_bias)

        cinput = np.dot(self.uW, node.inp_vec) + np.dot(self.uU, node.h_tilda)
        node.cell_input = np.tanh(cinput + self.u_bias)

        node.cell_state = i_gate * node.cell_input

        node.f_gates = {}
        for child in children:
            emb_vec = child.embedding
            f_gate = np.dot(self.fW, node.inp_vec) + np.dot(self.fU, emb_vec)
            node.f_gates[child.idx] = self.sigmoid(f_gate + self.f_bias)
            node.cell_state += f_gate * child.cell_state

        node.embedding = node.o_gate * np.tanh(node.cell_state)
        node.computed = True

    def update_word_embeddings(self):
        '''Use node gradients to update the word embeddings at each node.'''
        for node in self.tree:
            try:
                self.vectors[node.lower_] -= self.rate * node.inp_grad
            except KeyError:
                pass

    def update_weights(self):
        '''Use gradients to update all of the LSTM weights'''
        # update input gate weights
        self.iW -= self.rate * self.diW
        self.iU -= self.rate * self.diU
        self.i_bias -= self.rate * self.i_bias_grad

        # update forget gate weights
        self.fW -= self.rate * self.dfW
        self.fU -= self.rate * self.dfU
        self.f_bias -= self.rate * self.f_bias_grad

        # update output gate weights
        self.oW -= self.rate * self.doW
        self.oU -= self.rate * self.doU
        self.o_bias -= self.rate * self.o_bias_grad

        # update cell input weights
        self.uW -= self.rate * self.duW
        self.uU -= self.rate * self.duU
        self.u_bias -= self.rate * self.u_bias_grad

    def forward_pass(self, sentence):
        '''Compute activations for every node in the computational graph
        generated from a dependency parse of the provided sentence.'''
        self.tree = [TokenWrapper(token) for token in self.parser(sentence)]
        self.compute_embeddings()
        self.reset_comp_graph()

    def backward_pass(self, error_grad, rate=0.35):
        '''Compute gradients for every weight matrix and input word vector
        used when computing activations in accordance with the comp graph.'''
        self.diW = np.zeros_like(self.iW)
        self.doW = np.zeros_like(self.oW)
        self.dfW = np.zeros_like(self.fW)
        self.duW = np.zeros_like(self.uW)

        self.diU = np.zeros_like(self.iU)
        self.doU = np.zeros_like(self.oU)
        self.dfU = np.zeros_like(self.fU)
        self.duU = np.zeros_like(self.uU)

        self.i_bias_grad = np.zeros_like(self.i_bias)
        self.f_bias_grad = np.zeros_like(self.f_bias)
        self.o_bias_grad = np.zeros_like(self.o_bias)
        self.u_bias_grad = np.zeros_like(self.u_bias)

        self._set_root_gradient(error_grad)
        self.rate = rate
        self.compute_gradients()
        self.update_weights()
        self.update_word_embeddings()

    def get_children(self, node):
        '''Returns all nodes that are children of the provided node.'''
        children = []
        for other_node in self.tree:
            if other_node.idx in [child.idx for child in node.children]:
                children.append(other_node)

        return children

    def get_parent(self, node):
        '''Get the node that is the parent of the supplied node'''
        for other_node in self.tree:
            if other_node.idx == node.head.idx:
                return other_node

    def has_children(self, node):
        '''Check if node has children, return False for leaf nodes.'''
        return bool(node.children)

    def get_root_embedding(self):
        '''Returns the embedding for the root node in the tree.'''
        for node in self.tree:
            if node.head.idx == node.idx:
                return node.embedding

    def _set_root_gradient(self, grad):
        '''Set the error gradient on the root node in the comp graph.'''
        for node in self.tree:
            if node.head.idx == node.idx:
                node.top_grad = grad
                c_grad = node.o_gate * node.top_grad
                node.c_grad = c_grad
                self.grad_update(node, node)
