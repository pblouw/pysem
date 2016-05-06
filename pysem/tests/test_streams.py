import os
import re
import types
import pytest

from pysem.streams import StreamGenerator

corpus_path = os.getcwd() + '/tests/corpora'


def test_stream_build():
    wikitext = StreamGenerator(corpus_path)
    streams = wikitext.build_streams(n_streams=1, n_files=1)

    assert isinstance(streams.pop(), types.GeneratorType)

    with pytest.raises(ValueError) as execinfo:
        streams = wikitext.build_streams(n_streams=1, n_files=3)

    assert execinfo.value.message == 'sample larger than population'

def test_stream():
    wikitext = StreamGenerator(corpus_path)
    streams = wikitext.build_streams(n_streams=2, n_files=1)

    for stream in streams:
        assert isinstance(stream.next(), str)

def test_preprocess():
    wikitext = StreamGenerator(corpus_path)
    streams = wikitext.build_streams(n_streams=1, n_files=1)

    sens = list(streams.pop())
    text = ' '.join(sens)

    pattern = re.compile(r"<.*>")
    assert not pattern.findall(text)
    
    pattern = re.compile(r"\n")
    assert not pattern.findall(text)