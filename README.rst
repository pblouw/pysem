.. image:: https://travis-ci.com/pblouw/pysem.svg?token=xPcEs43jAf4HFvdb6WM7&branch=master
  :target: https://travis-ci.org/pblouw/pysem
  :alt: Travis-CI build status

.. image:: https://codecov.io/gh/pblouw/pysem/branch/master/graph/badge.svg?token=FdhAk7v0S0
  :target: https://codecov.io/gh/pblouw/pysem

*******************************************
PySem: Natural Language Semantics in Python
*******************************************

A suite of tools for doing NLP with distributed representations, with a
specific focus on natural language inference tasks such as question answering
and recognizing textual entailment. These tools range from implementations of
existing algorithms and techniques, to novel algorithms for generating sentences
that follow from a given sentence.


Features:
---------

* Text Corpora: Streams of Wikipedia articles or sentences with options for preprocessing and caching.
  Tools for using the Stanford Natural Language Inference corpus for recognizing textual entailment. 

* Word Embeddings: random indexing with options for encoding information concerning word order and 
  dependency syntax. Fully parallelized with Python's multiprocessing module.

* Standard ML Tools: logistic regression, multilayer perceptron.

* Neural Networks: standard RNNs and TreeRNNs (i.e. recursive neural networks), along with a TreeRNN
  variant that learns to encode a "Holographic Reduced Representation" (Plate 2003) of an input sentence. 

* Generative Modelling: experimental models that (a) learn the weights in a "decoding" TreeRNN to generate an 
  embedding for each node in the tree, and (b) learn weights for a TreeRNN that generates both structure and
  and content simultaneously. 

Examples:
---------

In the examples directory, there are Jupyter notebooks illustrating the creation of:

* embedding models from Wikipedia text

* classification models for predicting inferential relations betweeen sentences

* generative models for generating sentences that are entailed by a given sentence. 


Installation
-------------

Pysem requires Python 3, mostly to support effective multiprocessing. To install other requirements and library itself, do the following:

``
pip install -r requirements.txt

python setup.py develop --user
``

The --user flag can be omitted if you are using virtualenv or something equivalent. 


Testing:
--------

All of the machine learning and neural network models are tested with comprehensive gradient checks to
ensure that they are implemented correctly. Most other components of the library are tested as well.
Testing is performed using py.test.