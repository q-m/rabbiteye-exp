#!/usr/bin/env python
# usage: knn_test.py <image> <k>
import sys

from src import *

def print_nearest(knn, img, k):
    ret, result, neighbours, dist = knn.find_nearest(img, k)
    print ret, result, neighbours, dist
    if ret is not None: print knn.labels[ret]

knn_header = KNN()
knn_header.load('knn_data.header.npz')

knn_name = KNN()
knn_name.load('knn_data.nam.npz')

knn_unit = KNN()
knn_unit.load('knn_data.unt.npz')

knn_value = KNN()
knn_value.load('knn_data.val.npz')

f = File(sys.argv[1])
k = int(sys.argv[2])

print knn_header.find(f.header(), k=k)
for row_img in f.rows():
    row = Row(row_img)
    name = knn_name.find(row.name_img(), k=k)
    unit = knn_unit.find(row.unit_img(), k=k)

    number = Number(row.value_img())
    snumber = []
    for j, digit_img in enumerate(number.digit_imgs()):
        snumber.append(knn_value.find(digit_img, k=k))

    print "%s: %s %s"%(name, ''.join(snumber), unit)
