#!/usr/bin/env python
# usage: knn_test_single.py <image> <k>
import sys
import cv2

from src import KNN

def print_nearest(knn, img, k):
    ret, result, neighbours, dist = knn.find_nearest(img, k)
    print ret, result, neighbours, dist
    if ret is not None: print knn.labels[ret]

knn = KNN()
knn.load('knn_data.npz')

img = cv2.imread(sys.argv[1], 0)
k = int(sys.argv[2])

print_nearest(knn, img, k)

