#!/usr/bin/env python3
#
# Reuters newswire topic classification prediction example
#
import json
from lib.featurizer import Featurizer
from lib.categorizer import Categorizer
from lib.classifier import Classifier

testdata = [
    {"text": "local chamber of commerce takes action on legislation"},
    {"text": "consumption of food is estimated to have increased twofold in the past year"},
    {"text": "banking company offers settlement in long running financial case"},
    {"text": "negotiations on foreign affairs between china and australia enter into a new phase"}
]

f = Featurizer.load('model/reuters')
x = f.transform(testdata)
x_inv = f.transform_inv(x)
del f

m = Classifier.load('model/reuters')
y = m.predict(x)
del m

c = Categorizer.load('model/reuters_y')
y = c.transform_inv(y)

for i, line in enumerate(testdata):
    print("Input:", json.dumps(line))
    print("Features:", json.dumps(x_inv[i]))
    print("Prediction:", y[i])
    # too bad we don't know the category name - https://stackoverflow.com/questions/45138290

