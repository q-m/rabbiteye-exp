#!/usr/bin/env python
#
# Splits all nutrient tables into small images for training
#
import os
import glob
import numpy
import cv2

from src import *

for filename in glob.iglob('imgs/*.png'):
    f = File(filename)
    basepath = 'train/' + os.path.splitext(os.path.basename(filename))[0]
    try:
        cv2.imwrite(basepath + '.header.png', f.header())
        for i, row in enumerate(f.rows()):
            cv2.imwrite(basepath + '.%02d.nam.png'%i, row[:, :244])
            cv2.imwrite(basepath + '.%02d.val.png'%i, row[:, 244:])
    except ValueError:
        print 'Skipping: ' + filename
        pass
