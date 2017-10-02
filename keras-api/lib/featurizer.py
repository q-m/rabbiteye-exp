#!/usr/bin/env python3
#
# Feature extraction
#
# When run as a program, expects jsonlines on stdin, outputs jsonlines with features.
# A classifier using this featurizer would use the matrix returned, but this program
# turns the matrix back into words, so that you have something readable.
# In this reuters newswire classification example, the jsonlines are in the form:
#
#    { "text": "discovery sheds new light on the nature of dark matter" }
#
# The output for this line would be something like:
#
#    ["discovery", "sheds", "new", "light", "on", "the", "nature", "of", "dark", "matter"]
#
# Limiting the number of words, one would get, for example:
#
#    $ python3 featurizer.py -m 5
#    { "text": "A new day is dawning." }
#    { "text": ">> A newer day is coming! <<" }
#    ^D
#    ["a", "day", "is", "new"]
#    ["a", "day", "is"]
#
# @todo process line-by-line instead of loading everything in memory (two-pass)
# @todo remove stop words
#
import json
import numpy as np
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer

class Featurizer:

    max_words = None
    tokenizer = None

    def __init__(self, max_words=1000):
        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words=max_words)

    def fit_transform(self, data):
        texts = [l['text'] for l in data]
        self.tokenizer.fit_on_texts(texts)
        # remove words that cross the max_words limit
        self.tokenizer.word_index = {k: v for k, v in self.tokenizer.word_index.items() if v <= self.max_words}
        return self.transform(data)

    def transform(self, data):
        texts = [l['text'] for l in data]
        return self.tokenizer.texts_to_matrix(texts, mode='binary')

    def transform_inv(self, m):
        index = {v: k for k, v in self.tokenizer.word_index.items()} # word index by id
        return [[index.get(i) for i in np.nonzero(line)[0] if i in index] for line in m]

    def save(self, filepath):
        with open(filepath + '_word_index.json', 'w') as f:
            f.write(json.dumps(self.tokenizer.word_index))

    @classmethod
    def load(cls, filepath):
        with open(filepath + '_word_index.json', 'r') as f:
            word_index = json.load(f)
            c = cls(max_words=len(word_index))
            c.tokenizer.word_index = word_index
            return c


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Extracts features from jsonlines input.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l', '--load', type=str, help='load word index from file (omit _word_index.json)')
    group.add_argument('-s', '--save', type=str, help='save word index to file (omit _word_index.json)')
    parser.add_argument('-m', '--max-words', type=int, help='maximum number of words (default 1000)', default=1000)
    parser.add_argument('-f', '--file', type=str, help='load features from file instead of standard input')
    args = parser.parse_args()

    data = []
    if args.file:
        with open(args.file, 'r') as f:
            for line in f:
                data.append(json.loads(line.rstrip()))
    else:
        for line in sys.stdin:
            data.append(json.loads(line.rstrip()))

    if args.load:
        featurizer = Featurizer.load(args.load)
        features = featurizer.transform(data)
    else:
        featurizer = Featurizer(max_words=args.max_words)
        features = featurizer.fit_transform(data)
        if args.save: featurizer.save(args.save)

    del data
    output = featurizer.transform_inv(features)

    for line in output:
        print(json.dumps(line))
