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

* Text Corpora: Wikipedia article and sentence streaming, with options for preprocessing and caching.
  Tools for using the Stanford Natural Language Inference corpus for recognizing textual entailment, along with the Sentences Involving Compositional Knowledge
  corpus.

* Word Embeddings: a random indexing implementation with options for encoding
  information concerning word order and dependency syntax. Fully parallelized with Python's multiprocessing module.

* Standard ML Tools: logistic regression, multilayer perceptron.

* Neural Networks: standard RNNs and TreeRNNs (i.e. recursive neural networks), along with a TreeRNN
  variant that learns to encode a "Holographic Reduced Representation" (Plate 2003) of an input sentence. WIP LSTM extensions of these models.

* Generative Modelling: experimental models that (a) learn the weights in a 
  "decoding" TreeRNN to generate an embedding for each node in the tree, and (b) learn weights for a TreeRNN that generates both structure and
  and content simultaneously. 

Examples:
---------

In the examples directory, there are Jupyter notebooks illustrating the creation of:

* unsupervised models that learn word embeddings from Wikipedia text

* classification models for predicting inferential relations betweeen sentences

* generative models for generating sentences that are entailed by a given 
  sentence. 

Note that to you will need to download a dump of wikipedia articles and preprocess them using `this tool <https://github.com/attardi/wikiextractor>`_. You will also need to change the path in the notebook to point to where you have saved this dump locally. For the generative modelling example, the pretrained model parameters used in the notebook can be found `here <https://drive.google.com/open?id=0BxRAh6Eg1us4SVRkWWxJMXhWTDg>`_.

Installation
-------------

Pysem requires Python 3.5, mostly to support effective multiprocessing. For installation, it is easist to use the `Anaconda <https://www.continuum.io/downloads>`_ Python distribution to create a conda environment as follows. Run these commands from inside the cloned repository:

.. code:: shell

  conda env create
  source activate pysem
  python -m spacy.en.download
  python -m nltk.downloader stopwords punkt
  python setup.py develop


The first command creates an environment called `pysem` that includes all the needed dependencies, while the second command activates it. The next commands download data for doing parsing and tokenizing. The final command install this library in the environment. You can verify that the installation was successful with the following command:

.. code:: shell

  py.test

To leave the environment and (optionally) delete it, do the following:

.. code:: shell

  source deactivate
  conda env remove -n pysem


Testing:
--------

All of the machine learning and neural network models are tested with comprehensive gradient checks to ensure that they are implemented correctly. The library is currently only tested on Python 3.5