import numpy as np
from pysem.utils.spacy import TokenWrapper
from pysem.networks import RecursiveModel


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
        self.i_bias = np.zeros(self.dim).reshape((self.dim, 1))

        # initialize forget gate weights
        self.fW = self.random_weights(dim)
        self.fU = self.random_weights(dim)
        self.f_bias = np.zeros(self.dim).reshape((self.dim, 1))

        # initialize output gate weights
        self.oW = self.random_weights(dim)
        self.oU = self.random_weights(dim)
        self.o_bias = np.zeros(self.dim).reshape((self.dim, 1))

        # initialize cell input weights
        self.uW = self.gaussian_id(dim)
        self.uU = self.gaussian_id(dim)
        self.u_bias = np.zeros(self.dim).reshape((self.dim, 1))

        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

    def reset_comp_graph(self):
        '''Flag all nodes in the graph as being uncomputed.'''
        for node in self.tree:
            node.computed = False

    def clip_gradient(self, node, clipval=5):
        '''Clip a large gradient so that its norm is equal to clipval.'''
        norm = np.linalg.norm(node.gradient)
        if norm > clipval:
            node.gradient = (node.gradient / norm) * clipval

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
                self.clip_gradient(parent)

                c_grad = node.o_gate * parent.gradient
                c_grad += parent.c_grad * parent.f_grad
                node.c_grad = c_grad

                # gradients for output gate
                o_grad = node.cell_state * parent.gradient
                node.o_grad = o_grad * self.sigmoid_grad(node.o_gate)

                self.doW += np.dot(node.o_grad, node.inp_vec.T)
                self.doU += np.dot(node.o_grad, node.h_tilda.T)

                # gradients for input gate
                i_grad = c_grad * node.cell_input
                node.i_grad = i_grad * self.sigmoid_grad(node.i_gate)

                self.diW += np.dot(node.i_grad, node.inp_vec.T)
                self.diU += np.dot(node.i_grad, node.h_tilda.T)

                # gradients for forget gate
                csum = sum([n.cell_state for n in self.get_children(node)])
                f_grad = csum * c_grad
                node.f_grad = f_grad * self.sigmoid_grad(node.f_gate)

                self.dfW += np.dot(node.f_grad, node.inp_vec.T)
                self.dfU += np.dot(node.f_grad, node.h_tilda.T)

                # gradients for cell input
                ci_grad = node.i_gate * c_grad
                node.ci_grad = ci_grad * self.tanh_grad(node.cell_input)

                self.duW += np.dot(node.ci_grad, node.inp_vec.T)
                self.duU += np.dot(node.ci_grad, node.h_tilda.T)

                # gradients for biases
                self.i_bias_grad += node.i_grad
                self.f_bias_grad += node.f_grad
                self.o_bias_grad += node.o_grad
                self.u_bias_grad += node.ci_grad

                inp_grad = np.zeros_like(c_grad)
                emb_grad = np.zeros_like(c_grad)

                inp_grad += np.dot(self.diW.T, node.i_grad)
                inp_grad += np.dot(self.dfW.T, node.f_grad)
                inp_grad += np.dot(self.doW.T, node.o_grad)
                inp_grad += np.dot(self.duW.T, node.ci_grad)

                emb_grad += np.dot(self.diW.T, node.i_grad)
                emb_grad += np.dot(self.dfW.T, node.f_grad)
                emb_grad += np.dot(self.doW.T, node.o_grad)
                emb_grad += np.dot(self.duW.T, node.ci_grad)

                node.gradient = emb_grad
                node.computed = True

        if all([node.computed for node in self.tree]):
            return
        else:
            self.compute_gradients()

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

        for child in children:
            emb_vec = child.embedding
            f_gate = np.dot(self.fW, node.inp_vec) + np.dot(self.fU, emb_vec)
            child.f_gate = self.sigmoid(f_gate + self.f_bias)
            node.cell_state += f_gate * child.cell_state

        node.embedding = node.o_gate * np.tanh(node.cell_state)
        node.computed = True

    def forward_pass(self, sentence):
        '''Compute activations for every node in the computational graph
        generated from a dependency parse of the provided sentence.'''
        self.tree = [TokenWrapper(token) for token in self.parser(sentence)]
        self.compute_embeddings()
        self.reset_comp_graph()

    def backward_pass(self, error_grad, rate=0.35):
        '''Compute gradients for every weight matrix and input word vector
        used when computing activations in accordance with the comp graph.'''
        self._set_root_gradient(error_grad)
        self.rate = rate

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

        self.compute_gradients()
        # self.update_weights()
        # self.update_word_embeddings()

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
                node.gradient = grad
                node.c_grad = np.zeros_like(grad)
                node.f_grad = np.zeros_like(grad)
                node.computed = True
