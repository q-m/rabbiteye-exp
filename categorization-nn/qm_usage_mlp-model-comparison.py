#!/usr/bin/env python3
#
# Testing out different models for category classification.
# You can set METHOD=b4 or another to choose the model.
#
# To run all, you can execute something like:
#
#     for METHOD in a b b2 b4 b5 b6 c; do
#       python3 qm_usage_mlp-model-comparison.py | tee out/model-comparison-x-$METHOD.log
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

def featurize(product):
    words = []
    words.extend(preprocess_text(product.get('name', None)))
    words.extend(preprocess_text(product.get('brand_name', None)))
    words.extend(preprocess_text(product.get('ingredients', [None])[0]))  # first ingredient
    return words

data = data_features(jsonlines(filename_source_data), featurize)

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
xm_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape', xm_train.shape, 'x_test shape', xm_test.shape, 'y_train shape', y_train.shape, 'y_test shape', y_test.shape)


## Model 1 (currently in use)

if method == 'a':
    model1 = Sequential()
    model1.add(Dense(512, input_shape=(max_words,)))
    model1.add(Activation('relu'))
    model1.add(Dropout(0.5))
    model1.add(Dense(num_classes))
    model1.add(Activation('softmax'))

    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
    model1.summary()

    history = model1.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)

    score = model1.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)

# ## Model 2: a deeper network

elif method == 'b':
    model2 = Sequential()
    model2.add(Dense(int(max_words/2), input_shape=(max_words,), activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(int(max_words/4), activation='relu'))
    model2.add(Dropout(0.3))
    if num_classes < max_words/8:
        model2.add(Dense(int(max_words/8), activation='relu'))
        model2.add(Dropout(0.3))
    if num_classes < max_words/16:
        model2.add(Dense(int(max_words/16), activation='relu'))
        model2.add(Dropout(0.3))
    model2.add(Dense(num_classes, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
    model2.summary()

    history = model2.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)

    score = model2.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)

elif method == 'b2':
    model2 = Sequential()
    model2.add(Dense(int(max_words/2), input_shape=(max_words,), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(int(max_words/4), activation='relu'))
    model2.add(Dropout(0.2))
    if num_classes < max_words/8:
        model2.add(Dense(int(max_words/8), activation='relu'))
        model2.add(Dropout(0.2))
    if num_classes < max_words/16:
        model2.add(Dense(int(max_words/16), activation='relu'))
        model2.add(Dropout(0.2))
    model2.add(Dense(num_classes, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
    model2.summary()

    history = model2.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)

    score = model2.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)

elif method == 'b3':
    model2 = Sequential()
    model2.add(Dense(int(max_words*0.8), input_shape=(max_words,), activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(int(max_words*0.6), activation='relu'))
    model2.add(Dropout(0.3))
    if num_classes < max_words*0.4:
      model2.add(Dense(int(max_words*0.4), activation='relu'))
      model2.add(Dropout(0.3))
    if num_classes < max_words*0.2:
        model2.add(Dense(int(max_words*0.2), activation='relu'))
        model2.add(Dropout(0.3))
    if num_classes < max_words*0.05:
        model2.add(Dense(int(max_words*0.05), activation='relu'))
        model2.add(Dropout(0.3))
    model2.add(Dense(num_classes, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
    model2.summary()

    history = model2.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs*2, validation_split=0.2, verbose=verbose)

    score = model2.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)

elif method == 'b4':
    model2 = Sequential()
    model2.add(Dense(int(max_words/2), input_shape=(max_words,), activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(int(max_words/4), activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(int(max_words/8), activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(int(max_words/16), activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(int(max_words/32), activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(int(num_classes*0.4), activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(int(num_classes*0.8), activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(num_classes, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
    model2.summary()

    history = model2.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs*3, validation_split=0.2, verbose=verbose)

    score = model2.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)

elif method == 'b5':
    model2 = Sequential()
    model2.add(Dense(int(max_words/2), input_shape=(max_words,), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(int(max_words/4), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(int(max_words/8), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(int(max_words/16), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(int(max_words/32), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(int(num_classes*0.4), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(int(num_classes*0.8), activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(num_classes, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
    model2.summary()

    history = model2.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs*3, validation_split=0.2, verbose=verbose)

    score = model2.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)

elif method == 'b6':
    model2 = Sequential()
    model2.add(Dense(int(max_words/2), input_shape=(max_words,), activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(int(max_words/4), activation='relu'))
    model2.add(Dropout(0.3))
    if num_classes < max_words/8:
        model2.add(Dense(int(max_words/8), activation='relu'))
        model2.add(Dropout(0.3))
    if num_classes < max_words/16:
        model2.add(Dense(int(max_words/16), activation='relu'))
        model2.add(Dropout(0.3))
    model2.add(Dense(num_classes, activation='sigmoid'))

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'accuracy'])
    model2.summary()

    history = model2.fit(xm_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)

    score = model2.evaluate(xm_test, y_test, batch_size=batch_size, verbose=0)


## Model 3: using a pre-trained embedding

elif method == 'c':
    from gensim.models import KeyedVectors
    from keras.layers import Embedding
    from keras.preprocessing.sequence import pad_sequences

    EMBEDDING_DIM  = 300
    MAX_SEQ_LEN    =  15
    embedding_file = 'GoogleNews-vectors-negative300.bin'

    word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    xm3_train = pad_sequences(x_train, maxlen=MAX_SEQ_LEN)
    xm3_test  = pad_sequences(x_test, maxlen=MAX_SEQ_LEN)

    word_index = tokenizer.word_index
    nb_words = min(max_words, len(word_index)) + 1

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < nb_words and word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    embedding_layer = Embedding(embedding_matrix.shape[0], # or len(word_index) + 1
                                embedding_matrix.shape[1], # or EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQ_LEN,
                                trainable=False)


    from keras.models import Sequential
    from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
    from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

    model3= Sequential()
    model3.add(embedding_layer)
    model3.add(Dropout(0.2))
    model3.add(Conv1D(EMBEDDING_DIM, 3, padding='valid',activation='relu',strides=2))
    model3.add(Flatten())
    model3.add(Dropout(0.2))
    #model3.add(Dense(int(num_labels/2),activation='sigmoid'))
    model3.add(Dense(num_classes,activation='sigmoid'))
    model3.add(Dropout(0.2))
    model3.add(Dense(num_classes,activation='sigmoid'))

    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[recall, 'acc'])
    model3.summary()

    history = model3.fit(xm3_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=verbose)

    score = model3.evaluate(xm3_test, y_test, batch_size=batch_size, verbose=0)


## Summary

print(['score', 'recall', 'accuracy'])
print(score)
