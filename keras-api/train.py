#!/usr/bin/env python3
#
# Reuters newswire topic classification training example
#
from keras.datasets import reuters
from lib.featurizer import Featurizer
from lib.categorizer import Categorizer
from lib.classifier import Classifier

# load the reuters dataset
(x, y), _ = reuters.load_data(test_split=0, index_from=2)
word_index = reuters.get_word_index()

def dict_inv(d):
    '''Invert a dictionary'''
    return {v: k for k, v in d.items()}

def x2text(x, word_index_inv):
    '''Return text from an x vector and inverted word index'''
    words = [word_index_inv.get(i) for i in x]
    words = [w for w in words if w]
    return ' '.join(words)

# we use our own featurizer, so first reconstruct input text
word_index_inv = dict_inv(word_index)
texts = [{'text': x2text(a, word_index_inv)} for a in x]
del x, word_index, word_index_inv

# extract features, save results
f = Featurizer()
x = f.fit_transform(texts)
f.save('model/reuters')
del f

# build class mapping
m = Categorizer()
y = m.fit_transform(y)
m.save('model/reuters_y')
del m

# train the classifier, save results
c = Classifier(x.shape[1], y.shape[1])
c.train(x, y, epochs=12, batch_size=128)
c.save('model/reuters')
