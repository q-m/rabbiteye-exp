# Converts numerical array to matrix and vice versa.
#
import h5py
import numpy as np
import numpy_indexed as npi
from keras.utils import to_categorical

class Categorizer:

    def __init__(self):
        pass

    def fit_transform(self, y):
        self.mapping, ids = np.unique(y, return_inverse=True)
        return to_categorical(ids, num_classes=self.mapping.size)

    def transform(self, y):
        return npi.indices(self.mapping, y)

    def transform_inv(self, y):
        return self.mapping[np.argmax(y, axis=1)]

    def save(self, filepath):
        with h5py.File(filepath + '_categories.h5', 'w') as hf:
            hf.create_dataset('mapping', data=self.mapping)

    @classmethod
    def load(cls, filepath):
        c = cls()
        with h5py.File(filepath + '_categories.h5', 'r') as hf:
            c.mapping = hf['mapping'][:]
        return c
