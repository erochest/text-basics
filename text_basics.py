#!/usr/bin/env python3


"""Text analysis from basics."""


from collections import Counter
from datetime import datetime
import json
import os
import re

import numpy as np
from nltk.tokenize.casual import TweetTokenizer
# from matplotlib.mlab import PCA
# import matplotlib.pyplot as plt

INPUT_DIR = 'twitter/'
STOP_FILE = 'english.stopwords'


class Tweet(object):
    """A container for the data and actions we want with tweets."""

    def __init__(self, author, date, text):
        self.author = author
        self.date = date
        self.orig_text = text
        self.text = text

    @staticmethod
    def from_tweet(tweet):
        """Create a Tweet from a JSON object."""
        author = tweet['user_name']
        date = datetime.fromtimestamp(tweet['created_at'] / 1000)
        text = tweet['text']
        return Tweet(author, date, text)

    def over_text(self, func):
        """Replace self.text with the results of passing text to func."""
        self.text = func(self.text)
        return self.text

    def map_tokens(self, func):
        """
        Replace self.text with the results of passing each item in text to func.
        """
        text = []
        for token in self.text:
            text.append(func(token))
        self.text = text
        return text

    def __str__(self):
        return self.orig_text

    def __iter__(self):
        return iter(self.text)


def read_corpus(input_dir):
    """Read the input corpus and return a list of Tweet objects. """
    corpus = []

    for filename in os.listdir(input_dir):
        full_filename = os.path.join(input_dir, filename)

        with open(full_filename) as file_in:
            for line in file_in:
                if line.startswith('{'):
                    corpus.append(Tweet.from_tweet(json.loads(line)))

    return corpus


tokenizer = TweetTokenizer(reduce_len=True)
def tokenize(text):
    """Break a text into tokens."""
    return tokenizer.tokenize(text)


def normalize(token):
    """Normalize a token."""
    token = token.lower()
    if '/' in token:
        token = '#URL#'
    elif token.isdigit():
        token = '#NUM#'
    return token


def filter_tokens(tokens, stopwords):
    """
    Filter out tokens.

    * stopwords
    * word length
    * -frequency-
    """

    filtered = []

    for token in tokens:
        if token not in stopwords and len(token) > 1:
            filtered.append(token)

    return filtered


def ngrams(tokens, n):
    """Return lists of n-grams."""
    grams = []
    for i in range(len(tokens) - (n - 1)):
        grams.append(' '.join(tokens[i:i+n]))
    return grams


#  def collocates(tokens, n):
    #  """Return pairs of collocates."""


def frequencies(tokens):
    """Calculate frequencies for tokens."""
    return Counter(tokens)


def corpus_frequencies(corpus):
    """Aggregate all frequencies in a corpus."""
    counts = Counter()
    for tweet in corpus:
        counts.update(tweet.text.elements())
    return counts


def build_inverse_index(corpus):
    """This builds an inverse index based on the tokens in each item."""
    index = {}

    for doc in corpus:
        for token in doc:
            if token not in index:
                index[token] = []
            index[token].append(doc)

    return index


def make_token_dictionary(corpus):
    """Create a dictionary of tokens to vector positions."""
    token_dict = {}

    for doc in corpus:
        for token in doc:
            if token not in token_dict:
                token_dict[token] = len(token_dict)

    return token_dict


def vectorize(index, freqs):
    """Create a document vector from frequencies."""
    vector = [0.0] * len(index)
    for token, freq in freqs.items():
        i = index[token]
        vector[i] = freq
    return np.array(vector)


#  def vector_distance(vec_a, vec_b):
    #  """Return the Euclidean distance between two vectors."""


#  def cosine_similarity(vec_a, vec_b):
    #  """Return the cosine similarity between two vectors."""
    #  # https://en.wikipedia.org/wiki/Cosine_similarity


#  def kmeans(corpus, k):
    #  """Cluster the texts in a corpus."""


#  def graph_clusters(clusters, filename):
    #  """Generate a graph of clusters."""
    #  # http://matplotlib.org/examples/lines_bars_and_markers/scatter_with_legend.html


def main():
    """The main function. It all starts here."""
    corpus = read_corpus(INPUT_DIR)
    with open(STOP_FILE) as file_in:
        stopwords = set(
            normalize(token)
            for token in tokenize(file_in.read())
            )

    print(corpus[0].text)
    for tweet in corpus:
        tweet.over_text(tokenize)
        tweet.map_tokens(normalize)
        tweet.over_text(lambda text: filter_tokens(text, stopwords))
        # tweet.over_text(lambda text: ngrams(text, 2))
        tweet.over_text(frequencies)
    print(corpus[0].text)
    corpus_count = corpus_frequencies(corpus)

    token_count = 0
    for tweet in corpus:
        token_count += len(tweet.text)
    print('{} tokens in {} tweets'.format(token_count, len(corpus)))
    print('{} tokens/tweet'.format(token_count / len(corpus)))
    for word, freq in corpus_count.most_common(15):
        print('%15s %d' % (word, freq))

    type_count = len(corpus_count)
    token_count = sum(corpus_count.values())
    print('{} token/type ratio.'.format(token_count / type_count))

    #  index = build_inverse_index(corpus)
    #  for tweet in index.get('kim', []):
        #  print(str(tweet))

    token_dict = make_token_dictionary(corpus)
    matrix = [vectorize(token_dict, tweet.text) for tweet in corpus]


if __name__ == '__main__':
    main()
