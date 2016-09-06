

def dnn_accuracy(data, classifier, s1_dnn, s2_dnn):
    count = 0
    label_dict = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

    for sample in data:
        label = sample.label

        s1_dnn.forward_pass(sample.sentence1)
        s1_emb = s1_dnn.get_root_embedding()

        s2_dnn.forward_pass(sample.sentence2)
        s2_emb = s2_dnn.get_root_embedding()

        xs = np.concatenate((s1_emb, s2_emb))
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

class BagOfWords(object):

    def __init__(self, data, dim, pretrained=False):
        self.vocab = data.vocab
        self.train_data = [d for d in data.train_data if d.label != '-']
        self.dev_data = [d for d in data.dev_data if d.label != '-']
        self.test_data = [d for d in data.test_data if d.label != '-']

        self.dim = dim
        self.vectorizer = CountVectorizer(binary=True)
        self.vectorizer.fit(self.vocab)

        scale = 1 / np.sqrt(dim)
        size = (dim, len(vectorizer.get_feature_names()))
       
        if pretrained:
            vocab = vectorizer.get_feature_names()
            idx_lookup = {word: idx for idx, word in enumerate(vocab)}
            self.matrix = np.zeros((dim, len(vocab)))

            with open('pretrained_snli_embeddings.pickle', 'rb') as pfile:
                word2vec = pickle.load(pfile)

            for word in vocab:
                idx = idx_lookup[word]
                try:
                    emb = word2vec[word]
                except KeyError:
                    scale = 1 / np.sqrt(dim)
                    emb = np.random.normal(loc=0, scale=scale, size=dim)
                self.matrix[:, idx] = normalize(emb)
        else:
            self.matrix = np.random.normal(loc=0, scale=scale, size=size)

    def train