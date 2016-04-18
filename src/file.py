import numpy
import cv2

class File:
    def __init__(self, filename, cut=True, preprocess=True):
        self.filename = filename
        self.img = cv2.imread(filename, 0)
        if cut: self.img = self.img[1:-1, 1:-1]
        self.preprocess = preprocess
        self.__ypositions_cached = None

    def header(self):
        return self.__process(self.__row_img(0), invert=True)

    def rows(self):
        n = self.__ypositions().shape[0] - 1
        return map(self.__process, map(self.__row_img, range(1, n)))

    def __process(self, img, invert=False):
        if self.preprocess:
            img = cv2.resize(img, None, fx=3, fy=3)
            if invert:
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        return img

    def __ypositions(self):
        if self.__ypositions_cached is None:
            # detect lines
            img = self.img
            img = cv2.Canny(img, 20, 200)
            lines = cv2.HoughLinesP(img, 1, numpy.pi/180, 200, img.shape[1]*0.8, 5)
            # pick horizontal lines, and y-position only
            lines = lines[0, lines[0,:,1] == lines[0,:,3], 1]
            lines.sort()
            # add top and bottom line (just to be sure)
            lines2 = numpy.zeros(lines.shape[0]+2, dtype=numpy.uint32)
            lines2[1:-1] = lines
            lines[-1] = img.shape[0] - 1
            # now remove lines closer than 4 pixels to another
            lines2 = lines2[ lines2[1:]-lines2[:-1] > 3 ]
            # add bottom line (need to check above why it's not there)
            lines3 = numpy.zeros(lines2.shape[0]+1, dtype=numpy.uint32)
            lines3[0:-1] = lines2
            lines3[-1] = img.shape[0] - 1
            self.__ypositions_cached = lines3
        return self.__ypositions_cached

    def __row_img(self, i):
        ys = self.__ypositions()
        return self.img[ys[i]:ys[i+1]]

