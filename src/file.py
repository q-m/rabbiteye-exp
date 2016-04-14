import numpy
import cv2

class File:
    def __init__(self, filename, cut=True, preprocess=True):
        self.img = cv2.imread(filename, 0)
        self.cut = cut
        self.preprocess = preprocess

    def header(self):
        return self.__process(self.img[0:30], invert=True)

    def rows(self):
        return map(self.__process, numpy.vsplit(self.img[30:], 8))

    def __process(self, img, invert=False):
        if self.cut:
            img = img[1:-1, 1:-1]
        if self.preprocess:
            ret, img = cv2.threshold(img, 0, 255, (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) + cv2.THRESH_OTSU)
        return img

