#!/usr/bin/env python3


"""Text analysis from basics."""


from datetime import datetime
import json
import os
import re

# import numpy as np
from nltk.tokenize.casual import TweetTokenizer
# from matplotlib.mlab import PCA
# import matplotlib.pyplot as plt

INPUT_DIR = 'twitter/'
# STOP_FILE = 'english.stopwords'


class Tweet(object):
    """A container for the data and actions we want with tweets."""

    def __init__(self, author, date, text):
        self.author = author
        self.date = date
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


#  def normalize(token):
    #  """Normalize a token."""


#  def filter_tokens(tokens, stop_file):
    #  """
    #  Filter out tokens.

    #  * stopwords
    #  * word length
    #  * -frequency-
    #  """


#  def ngrams(tokens, n):
    #  """Return lists of n-grams."""


#  def collocates(tokens, n):
    #  """Return pairs of collocates."""


#  def frequencies(tokens):
    #  """Calculate frequencies for tokens."""


#  def make_token_dictionary(corpus):
    #  """Create a dictionary of tokens to vector positions."""


#  def vectorize(index, freqs):
    #  """Create a document vector from frequencies."""


#  def vector_distance(vec_a, vec_b):
    #  """Return the Euclidean distance between two vectors."""


#  def cosine_distance(vec_a, vec_b):
    #  """Return the cosine similarity between two vectors."""


#  def kmeans(corpus, k):
    #  """Cluster the texts in a corpus."""


#  def graph_clusters(clusters, filename):
    #  """Generate a graph of clusters."""


def main():
    """The main function. It all starts here."""
    corpus = read_corpus(INPUT_DIR)

    print(corpus[0].text)
    for tweet in corpus:
        tweet.over_text(tokenize)
    print(corpus[0].text)

    token_count = 0
    for tweet in corpus:
        token_count += len(tweet.text)
    print('{} tokens in {} tweets'.format(token_count, len(corpus)))
    print('{} tokens/tweet'.format(token_count / len(corpus)))


if __name__ == '__main__':
    main()
