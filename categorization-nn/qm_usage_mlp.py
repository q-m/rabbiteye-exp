'''Trains and evaluate a simple MLP on the Questionmark usage classification task.'''
from __future__ import print_function

import json
import numpy as np
import keras
from itertools import groupby
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

from keras_metrics import recall
from featurize import tokenize_dict

max_words = 20000
batch_size = 128
epochs = 25

base_filename = 'qm_usage_mlp.out'
filename_source_data = 'data/product_nuts_with_product_info.jsonl'
filename_removed_classes = base_filename + '.removed_classes.txt'
filename_model = base_filename + '.model.h5'

#---- extract features

def jsonlines(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    return data

print('Loading data...')
data = [tokenize_dict(p) for p in jsonlines(filename_source_data)]
data = [d for d in data if d['usage']] # remove entries without a class

# data is now a list of
#
#   {'id': 1463,
#    'product_id': 663720,
#    'tokens': <filter at 0x7fa4237de898>, # --> ['witte', 'champignons', 'fijn', '250g', 'BRN:Jumbo', 'ING:champignons']
#    'usage': 'Aardappelpuree, mix voor'}
#

x = ['|'.join(d['tokens']) for d in data]
y = [d['usage'] for d in data]
p = [d['product_id'] for d in data]

#---- integer features

y_labels = list(set(y))
y = [y_labels.index(l) for l in y]

tokenizer = Tokenizer(num_words=max_words, split='|', filters='', lower=False)
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)


#---- filter features
# @todo this can probably be rewritten much more beautifully using numpy arrays

# remove items that didn't get any features at all
yx = [[y[i], x, p[i]] for i,x in enumerate(x)]
yx = [d for d in yx if len(d[1]) > 0]
new_y = [d[0] for d in yx]
new_x = [d[1] for d in yx]
new_p = [d[2] for d in yx]
print("Removed %d items without features"%(len(x)-len(new_x)))
x, y, p = new_x, new_y, new_p
del yx, new_x, new_y, new_p

# remove usages with less than 3 products
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
print("removed %d classes (with %d items) not appearing often enough: %s"%(len(removed_classes), removed_item_count, filename_removed_classes))
with open(filename_removed_classes, 'w') as f:
    for c in removed_classes: f.write(y_labels[c] + '\n')
x, y, p = new_x, new_y, new_p
del yx, new_x, new_y, new_p

# remove duplicates (each combination of features may appear only once)
#   ultimately, choosing a common parent usage for conflicts would be best, I guess :)
keyfunc = lambda d: d[1]
yx = [[y[i], x, p[i]] for i,x in enumerate(x)]
yx = groupby(sorted(yx, key=keyfunc), key=keyfunc)
yx = [list(d[1])[0] for d in yx]
new_y = [d[0] for d in yx]
new_x = [d[1] for d in yx]
new_p = [d[2] for d in yx]
print("Removed %d duplicates"%(len(x)-len(new_x)))
x, y, p = new_x, new_y, new_p
del yx, new_x, new_y, new_p


#---- split into train and test set (without bubbles, yet)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #, random_state=0)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

del data
del x
del y


#---- keras

print('Vectorizing sequence data...')
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[recall, 'accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
model.save(filename_model)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print()
print('Test score:', score[0])
print('Test recall:', score[1])
print('Test accuracy:', score[2])

