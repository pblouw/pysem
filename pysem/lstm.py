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
        self.i_bias = np.zeros(dim)

        # initialize forget gate weights
        self.fW = self.random_weights(dim)
        self.fU = self.random_weights(dim)
        self.f_bias = np.zeros(dim)

        # initialize output gate weights
        self.oW = self.random_weights(dim)
        self.oU = self.random_weights(dim)
        self.o_bias = np.zeros(dim)

        # initialize cell input weights
        self.uW = self.gaussian_id(dim)
        self.uU = self.gaussian_id(dim)
        self.cell_bias = np.zeros(dim)

        self.pretrained_vecs(pretrained) if pretrained else self.random_vecs()

    def reset_comp_graph(self):
        '''Flag all nodes in the graph as being uncomputed.'''
        for node in self.tree:
            node.computed = False

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

    def update_node(self, node, children):
        '''
        '''
        h_tilda = sum([node.embedding for node in children])
        try:
            inp_vec = self.vectors[node.lower_]
        except KeyError:
            inp_vec = np.zeros(self.dim).reshape((self.dim, 1))

        i_gate = np.dot(self.iW, inp_vec) + np.dot(self.iU, h_tilda)
        i_gate = self.sigmoid(i_gate + self.i_bias)

        o_gate = np.dot(self.oW, inp_vec) + np.dot(self.oU, h_tilda)
        o_gate = self.sigmoid(o_gate + self.o_bias)

        cell_input = np.dot(self.uW, inp_vec) + np.dot(self.uU, h_tilda)
        cell_input = np.tanh(cell_input + self.cell_bias)

        node.cell_state = i_gate * cell_input

        for child in children:
            emb_vec = child.embedding
            f_gate = np.dot(self.fW, inp_vec) + np.dot(self.fU, emb_vec)
            f_gate = self.sigmoid(f_gate + self.f_bias)
            node.cell_state += f_gate * child.cell_state

        node.embedding = o_gate * np.tanh(node.cell_state)
        node.computed = True

    def forward_pass(self, sentence):
        '''Compute activations for every node in the computational graph
        generated from a dependency parse of the provided sentence.'''
        self.tree = [TokenWrapper(token) for token in self.parser(sentence)]
        self.compute_embeddings()
        self.reset_comp_graph()

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
