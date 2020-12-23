#!/usr/bin/env python3
"""Train.py

Derived from: https://colab.research.google.com/drive/1SYppYxzGamoaaVHXb1mItE9NF-nmfkJO
"""
import os

import tensorflow as tf
print(tf.__version__)

from config import *
from custom_model import new_trainer

### Remove cache files
#
# This needs to be done each time the input data changes, which would
# probably be on each run. Besides, it doesn't take that long to generate
# them.
#

if os.path.exists(DATA_DIR+'cache/detection_train_data.pkl') : 
   os.remove(DATA_DIR+'cache/detection_train_data.pkl')
if os.path.exists(DATA_DIR+'cache/detection_test_data.pkl') : 
   os.remove(DATA_DIR+'cache/detection_test_data.pkl')

### Train
#

trainer = new_trainer()
trainer.trainModel()
