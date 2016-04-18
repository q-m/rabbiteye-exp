#!/usr/bin/env python
#
# Splits all nutrient tables into small images for training
#
import os
import glob
import numpy
import cv2

from src import File, Row, Number

for filename in glob.iglob('imgs/*.png'):
    f = File(filename)
    basepath = 'train/' + os.path.splitext(os.path.basename(filename))[0]
    try:
        cv2.imwrite(basepath + '.header.png', f.header())
        for i, row_img in enumerate(f.rows()):
            row = Row(row_img)
            cv2.imwrite(basepath + '.%02d.nam.png'%i, row.name_img())
            cv2.imwrite(basepath + '.%02d.unt.png'%i, row.unit_img())
            number = Number(row.value_img())
            for j, digit_img in enumerate(number.digit_imgs()):
                cv2.imwrite(basepath + '.%02d.val.%1d.png'%(i,j), digit_img)
    except ValueError, e:
        print 'Skipping "%s": %s'%(filename, str(e))
        pass
