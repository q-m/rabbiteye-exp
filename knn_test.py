#!/usr/bin/env python
# usage: knn_test.py <image> <k>
import sys

from src import File, KNN

def print_nearest(knn, img, k):
    ret, result, neighbours, dist = knn.find_nearest(img, k)
    print ret, result, neighbours, dist
    if ret is not None: print knn.labels[ret]

knn = KNN()
knn.load('knn_data.npz')

f = File(sys.argv[1])
k = int(sys.argv[2])

print_nearest(knn, f.header(), k)
for row in f.rows():
  print_nearest(knn, row[:, :245], k)

