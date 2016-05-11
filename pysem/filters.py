
def snli_sentences(snli_stream):
    for snli_json in snli_stream:
        s1 = snli_json['sentence1']
        s2 = snli_json['sentence2']
        yield (s1, s2)


def snli_labels(snli_stream):
    for snli_json in snli_stream:
        yield snli_json['gold_label']
