# Product categorization using neural networks

After a first step in [applying machine learning for product categorization](../categorization-svm),
there were a lot of things to improve. Attempts to do so with SVM [did not really succeed](../categorization-svm-2).

Here we explore if neural networks can improve the performance of the usage classification.

## Dependencies

To run the examples, one needs [Python 3+](http://python.org/) with [Keras](https://keras.io/) and
[TensorFlow](https://www.tensorflow.org/). Make sure you also have [h5py](http://www.h5py.org/).

When [installing TensorFlow](https://www.tensorflow.org/install/), using an optimized version can
improve performance, see [this repo](https://github.com/yaroslavvb/tensorflow-community-wheels)'s
issues for builds. You may also need the [native protobuf package](https://www.tensorflow.org/install/install_linux#protobuf_pip_package_31).

## Pre-processing

[See here](../categorization-svm-2/README.md#Pre-processing).

## Experiments

### 1. Simple Multilayer Perceptron

The [reuters newswire classification example](https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py)
was used as a starting point. Conditions were tuned to be somewhat comparable with
[SVM (without bubbles)](../categorization-svm-2/cross_validation.ipynb), which had a 112k training
set and 28k test set, 29k features, resulting in a macro recall score of 0.79. The same featurization
and filtering was used.

The multilayer perceptron (MLP) had a 111k training set and 28k test set, 20k features (capped).
With 5 epochs, this resulted in a recall score of 0.78. Pretty comparable.

```
$ python3 qm_usage_mlp.py

Removed 48 items without features
removed 80 classes (with 122 items) not appearing often enough: qm_usage_mlp.out.removed_classes.txt
Removed 107624 duplicates

111032 train sequences
27758 test sequences
1322 classes

x_train shape: (111032, 25000)
x_test shape: (27758, 25000)
y_train shape: (111032, 1322)
y_test shape: (27758, 1322)

Train on 98895 samples, validate on 10989 samples
Epoch 1/5 - 132s - loss: 3.8216 - recall: 0.2019 - acc: 0.4053 - val_loss: 2.0277 - val_recall: 0.4566 - val_acc: 0.6518
Epoch 2/5 - 130s - loss: 1.4315 - recall: 0.5781 - acc: 0.7304 - val_loss: 1.1565 - val_recall: 0.6534 - val_acc: 0.7645
Epoch 3/5 - 132s - loss: 0.8105 - recall: 0.7289 - acc: 0.8253 - val_loss: 0.8888 - val_recall: 0.7291 - val_acc: 0.8048
Epoch 4/5 - 131s - loss: 0.5402 - recall: 0.8079 - acc: 0.8715 - val_loss: 0.7739 - val_recall: 0.7680 - val_acc: 0.8216
Epoch 5/5 - 132s - loss: 0.3972 - recall: 0.8535 - acc: 0.8993 - val_loss: 0.7243 - val_recall: 0.7928 - val_acc: 0.8325

Test score: 0.757754175588
Test recall: 0.783415841619
Test accuracy: 0.825931857926

$ du -sh qm_usage_mlp.out.model.h5
125M
```

For comparison:
* With 5 epochs, 10k features, (111k duplicates), recall/accuracy was 0.74/0.80, model size 67M.
* With 5 epochs, 5k features, (112k duplicates, recall/accuracy eas 0.69/0.77, model size 38M.
* With 25 epochs, 5k features, recall/accuracy was 0.80/0.81, model size still 38M.
* With 25 epochs, 20k features, recall/accuracy was 0.85/0.86, model size still 125M.

**later addition** On dataset `data-2` (from 20170706):
* With 5 epochs, 20k features (43k duplicates, 147k train, 37k test), recall/accuracy was 0.82/0.85, model size 126M.
* With 25 epochs, 5k features (66k duplicates, 126k train, 32k test), recall/accuracy was 0.81/0.82, model size 38M. Overfitting from about 10 epochs.
* With 25 epochs, 20k features (43k duplicates, 147k train, 37k test), recall/accuracy was 0.86/0.87, model size 126M. Overfitting from about 5 epochs.

### 2. Feature analysis

About half of the items were removed because they had equal features. This may be an indicator that
featurization is missing some distinguishing features, or that our training classes are ambiguous. This
may also be ok, because the dataset has items from different sources describing the same product.

Desired outputs:
- classes removed because of too little items (need more items or courser classes)
- items whose features belong to multiple classes (make manual classification unambiguous or expand features)

#### Different feature sets

Different feature selection methods are explored in [model_and_feature_search](model_and_feature_search.ipynb)
and [qm_usage_mlp-feature-comparison](qm_usage_mlp-feature-comparision.py). The conclusion was that brand,
model and first ingredient are enough, while adding more ingredients or even categories don't improve predictions.

More analysis could be done on:
- `max_words` parameter search (in progress)
- including (certain) nutrients
- incorporating description (probably need an embedding first)
- manually extract complex features (like alcohol percentage)
- feed different features into the model separately, to control what is (deemed) more important
- ...

#### Error analysis

Nevertheless, looking at where classification errors are happening (zooming in on certain classes) would
probably be good to start with, so as to see where features may be missing, or where our training set
has wrongly labelled classes.

### 3. Other models

While two-layer perceptron gave relatively good results, other neural network models could be an improvement.
A first exploration was done in [model_and_feature_search](model_and_feature_search.ipynb) (bottom) and
and [qm_usage_mlp-model-comparison](qm_usage_mlp-model-comparision.py). This showed no improvements for deeper
neural models, though a basic neural network did improve the learning rate.

Directions to explore are:
- creating embeddings for name, ingredients, description and using them
- ...

### 4. Embedding

At this moment, words are direct inputs for the neural network. This may be improved by using word embeddings,
which reduces the size of the network by 'combining' words to 'revelant feature vectors'. This process can use
_all_ data, not only labelled data, to discover 'structure' in the input.

A first step: experimenting with generating an embedding for ingredients, see
[gensim-embedding-ingredients](gensim-embedding-ingredients.ipynb).

Next steps would be:
- evaluating the quality of the resulting embedding
- creating embeddings for name and description
- using them in the classification model

