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

The multilayer perceptron (MLP) had a 111k training set and 28k test set, 20k features (capped),
resulting in a recall score of 0.76. Pretty comparable.

```
$ python3 qm_usage_mlp.py

Removed 59 items without features
removed 122 items of classes not appearing often enough
Removed 107778 duplicates

110900 train sequences
27725 test sequences
1322 classes

x_train shape: (110900, 20000)
x_test shape: (27725, 20000)
y_train shape: (110900, 1322)
y_test shape: (27725, 1322)

Epoch 1/5 - 103s - loss: 4.3827 - recall: 0.1274 - acc: 0.3386 - val_loss: 2.5874 - val_recall: 0.3464 - val_acc: 0.5865
Epoch 2/5 - 103s - loss: 1.8412 - recall: 0.4819 - acc: 0.6739 - val_loss: 1.4602 - val_recall: 0.5733 - val_acc: 0.7277
Epoch 3/5 - 113s - loss: 1.0640 - recall: 0.6523 - acc: 0.7830 - val_loss: 1.0621 - val_recall: 0.6722 - val_acc: 0.7872
Epoch 4/5 - 128s - loss: 0.7185 - recall: 0.7459 - acc: 0.8421 - val_loss: 0.8772 - val_recall: 0.7311 - val_acc: 0.8142
Epoch 5/5 - 115s - loss: 0.5317 - recall: 0.8062 - acc: 0.8751 - val_loss: 0.7817 - val_recall: 0.7671 - val_acc: 0.8317

Test score: 0.77819933308
Test recall: 0.764580703283
Test accuracy: 0.825103697061

$ du -sh qm_usage_mlp.h5
125M
```

For comparison:
Restricting to 10k features resulted in 111k duplicates, a recall/accuracy of 0.74/0.80 and model size of 67M.
Restricting to 5k features resulted in 112k duplicates, a recall/accuracy of 0.69/0.77 and model size 38M.

### 2. Feature analysis

About half of the items were removed because they had equal features. This may be an indicator that
featurization is missing some distinguishing features, or that our training classes are ambiguous.

TODO

### 3. Autoencoding

While there are about 250k items with classes, there are 850k items without. These can be used to
train an autoencoder to extract relevant features. Later these can be used as an input layer for classification.

TODO

### 4. Autoencoding + MLP

