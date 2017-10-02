# # The model
#
# train Reuters newswire classification model
# @see https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py
#
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation

class Classifier:
    '''
       Basic classifier model.

       Expects numerical input vectors as X and Y.
       Can train, evaluate and predict (just like a Keras model).
       Can also load and save the model.
    '''

    _model = None

    def __init__(self, xSize, ySize, _model=None):
        if _model:
            self._model = _model
        else:
            self._model = Sequential()
            self._model.add(Dense(512, input_shape=(xSize,)))
            self._model.add(Activation('relu'))
            self._model.add(Dropout(0.5))
            self._model.add(Dense(ySize))
            self._model.add(Activation('softmax'))
            self._model.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['accuracy'])

    def predict(self, x, **kwargs):
        return self._model.predict(x, **kwargs)

    def train(self, x, y, **kwargs):
        return self._model.fit(x, y, **kwargs)

    def evaluate(self, x, y, **kwargs):
        return self._model.evaluate(x, y, **kwargs)

    def save(self, filepath):
        # save model and weights
        with open(filepath + '_model.json', 'w') as f:
            f.write(self._model.to_json())
        self._model.save_weights(filepath + '_weights.h5')

    @classmethod
    def load(cls, filepath):
        # load model and weights
        with open(filepath + '_model.json', 'r') as f:
            model = model_from_json(f.read())
            # @todo verify xSize and ySize
        model.load_weights(filepath + '_weights.h5')
        return cls(None, None, _model=model)
