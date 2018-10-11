#!/usr/bin/env python3
#
# Testing out different feature selection methods for category classification.
# You can set METHOD=b or another to choose the feature selection method.
#
# To run all, you can execute something like:
#
#     for METHOD in a a2 b c d e f g h; do
#       python3 qm_usage_mlp-feature-comparison.py | tee out/feature-comparison-x-$METHOD.log
#     done
#
import os
import json
import numpy as np
import keras
import nltk
from itertools import groupby
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words  = int(os.getenv('MAX_WORDS',   5000))
batch_size = int(os.getenv('BATCH_SIZE',   128))
epochs     = int(os.getenv('EPOCHS',        15))

verbose              = int(os.getenv('VERBOSE', 1))
method               = os.getenv('METHOD', 'a')
filename_source_data = os.getenv('FILE', 'data/product_nuts_with_product_info.jsonl')

print("max_words=%d batch_size=%d epochs=%d method=%s file=%s"%(max_words, batch_size, epochs, method, filename_source_data))

nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words('dutch') + '''
  ca bijv bijvoorbeeld
  gemaakt aanbevolen
  belangrijk belangrijke heerlijk heerlijke handig handige dagelijks dagelijkse
  gebruik allergieinformatie bijdrage smaak hoeveelheid
'''.split()

# Important keras metric not present by default
from keras import backend as K
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


## Extract features

import re
from unidecode import unidecode

def jsonlines(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_single(s):
    if s is None: return None
    
    s = unidecode(s).strip()
    s = re.sub(r'[^A-Za-z0-9\'\s%]', '', s, flags=re.MULTILINE)
    s = s.lower()
    
    return s

def preprocess_text(s):
    if s is None: return []

    s = preprocess_single(s)
    words = s.split()
    words = [w for w in words if w not in STOPWORDS]
    return words

def tokenize_dict(j, method):
    d = {'id': j['id'], 'tokens': method(j)}
    if 'usage'      in j: d['usage']      = j['usage']
    if 'product_id' in j: d['product_id'] = j['product_id']

    return d

def data_features(data, method):
    data = [tokenize_dict(p, method) for p in data]
    data = [d for d in data if d['usage']] # remove entries without a class
    return data

# --------------

def featurize_a(product):
    words = []
    words.extend(preprocess_text(product.get('brand_name', None)))
    words.extend(preprocess_text(product.get('name', None)))
    return words

def featurize_a2(product):
    words = []
    words.extend(preprocess_text(product.get('name', None)))
    words.extend(preprocess_text(product.get('brand_name', None)))
    return words

def featurize_a3(product):
    words = []
    words.extend(['N:' + s for s in preprocess_text(product.get('name', None))])
    words.extend(['B:' + s for s in preprocess_text(product.get('brand_name', None))])
    return words

def featurize_b(product):
    words = []
    words.extend(preprocess_text(product.get('brand_name', None)))
    words.extend(preprocess_text(product.get('name', None)))
    words.extend(preprocess_text(product.get('ingredients', [None])[0]))  # first ingredient
    return words

def featurize_c(product):
    words = []
    words.extend([preprocess_single(s) for s in product.get('categories', [])])
    return words

def featurize_d(product):
    words = []
    for s in product.get('categories', []): words.extend(preprocess_text(s))
    return words

def featurize_e(product):
    words = []
    words.extend(preprocess_text(product.get('brand_name', None)))
    words.extend(preprocess_text(product.get('name', None)))
    for s in product.get('categories', []): words.extend(preprocess_text(s))
    return words

def featurize_f(product):
    words = []
    words.extend(preprocess_text(product.get('brand_name', None)))
    words.extend(preprocess_text(product.get('name', None)))
    words.extend(preprocess_text(product.get('ingredients', [None])[0]))  # first ingredient
    for s in product.get('categories', []): words.extend(preprocess_text(s))
    return words

def featurize_g(product):
    words = []
    for s in product.get('ingredients', []): words.extend(preprocess_text(s))
    return words

def featurize_h(product):
    words = []
    words.extend(preprocess_text(product.get('brand_name', None)))
    words.extend(preprocess_text(product.get('name', None)))
    for s in product.get('categories', []): words.extend(preprocess_text(s))
    for s in product.get('ingredients', []): words.extend(preprocess_text(s))
    return words

def featurize_h2(product):
    words = []
    words.extend(['B:' + s for s in preprocess_text(product.get('brand_name', None))])
    words.extend(['N:' + s for s in preprocess_text(product.get('name', None))])
    for s in product.get('categories', []): words.extend(['C:' + s for s in preprocess_text(s)])
    for s in product.get('ingredients', []): words.extend(['I:' + s for s in preprocess_text(s)])
    return words

def featurize_i(product):
    words = []
    words.extend(preprocess_text(product.get('brand_name', None)))
    words.extend(preprocess_text(product.get('name', None)))
    words.extend(preprocess_text(product.get('ingredients', [None])[0]))  # first ingredient
    for s in product.get('categories', []): words.extend(preprocess_text(s))
    return words

def featurize_i2(product):
    words = []
    words.extend(['B:' + s for s in preprocess_text(product.get('brand_name', None))])
    words.extend(['N:' + s for s in preprocess_text(product.get('name', None))])
    words.extend(['I:' + s for s in preprocess_text(product.get('ingredients', [None])[0])])  # first ingredient
    for s in product.get('categories', []): words.extend(['C:' + s for s in preprocess_text(s)])
    return words

featurize_methods = {
    'a': featurize_a,
    'a2': featurize_a2,
    'a3': featurize_a3,
    'b': featurize_b,
    'c': featurize_c,
    'd': featurize_d,
    'e': featurize_e,
    'f': featurize_f,
    'g': featurize_g,
    'h': featurize_h,
    'h2': featurize_h2,
    'i': featurize_i,
    'i2': featurize_i2,
}
    
data = data_features(jsonlines(filename_source_data), featurize_methods[method])

sample_id = 81
print("Sample idx %d with label '%s': %s"%(sample_id, data[sample_id]['usage'], data[sample_id]['tokens']))

## Integer features
y = [d['usage'] for d in data]
y_labels = list(set(y))
y = [y_labels.index(l) for l in y]

x = ['|'.join(d['tokens']) for d in data]
tokenizer = Tokenizer(num_words=max_words, split='|', filters='', lower=False)
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
p = [d['product_id'] for d in data]

## Filter features

# remove items that didn't get any features at all
def filter_nofeat(x, y, p):
    yx = [[y[i], x, p[i]] for i, x in enumerate(x)]
    yx = [d for d in yx if len(d[1]) > 0]
    new_y = [d[0] for d in yx]
    new_x = [d[1] for d in yx]
    new_p = [d[2] for d in yx]
    print("Removed %d items without features"%(len(x)-len(new_x)))
    return (new_x, new_y, new_p)

x, y, p = filter_nofeat(x, y, p)

# remove usages with less than 3 products
def filter_little_products(x, y, p):
    keyfunc = lambda d: d[0]
    yx = [[y[i], x, p[i]] for i,x in enumerate(x)]
    new_x, new_y, new_p = [], [], []
    for cur_y, cur_yx in groupby(sorted(yx, key=keyfunc), key=keyfunc):
        cur_yx = list(cur_yx)
        if len(cur_yx) >= 3:
            new_y.extend([d[0] for d in cur_yx])
            new_x.extend([d[1] for d in cur_yx])
            new_p.extend([d[2] for d in cur_yx])
    removed_item_count = len(y) - len(new_y)
    removed_classes = list(set(y) - set(new_y))
    print("removed %d classes (with %d items) not appearing often enough"%(len(removed_classes), removed_item_count))
    return (new_x, new_y, new_p)

x, y, p = filter_little_products(x, y, p)

# remove duplicates (each combination of features may appear only once)
#   ultimately, choosing a common parent usage for conflicts would be best, I guess :)
def filter_dups(x, y, p):
    keyfunc = lambda d: d[1]
    yx = [[y[i], x, p[i]] for i,x in enumerate(x)]
    yx = groupby(sorted(yx, key=keyfunc), key=keyfunc)
    yx = [list(d[1])[0] for d in yx]
    new_y = [d[0] for d in yx]
    new_x = [d[1] for d in yx]
    new_p = [d[2] for d in yx]
    print("Removed %d duplicates"%(len(x)-len(new_x)))
    return (new_x, new_y, new_p)

x, y, p = filter_dups(x, y, p)


## Split into train and test sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #, random_state=0)
num_classes = np.max(y) + 1

print(len(x_train), 'train sequences', len(x_test), 'test sequences', num_classes, 'classes')

## Vectorize sequence data

xm_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
xm_test =tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape', xm_train.shape, 'x_test shape', xm_test.shape, 'y_train shape', y_train.shape, 'y_test shape', y_test.shape)

## Model 2: a deeper network

model = Sequential()
model.add(Dense(int(max_words/2), input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(int(max_words/4), activation='relu'))
model.add(Dropout(0.3))
if num_classes < max_words/8:
    model.add(Dense(int(max_words/8), activation='relu'))
    model.add(Dropout(0.3))
if num_classes < max_words/16:
    model.add(Dense(int(max_words/16), activation='relu'))
    model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
model.summary()

h = model.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)

score = model.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)
print(['score', 'recall', 'accuracy'])
print(score)

