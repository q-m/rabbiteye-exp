import numpy
import cv2

# Single row in nutrient table
class Row:
    def __init__(self, img):
        self.img = img
        self.__segments_cached = None

    # name is just left half
    def name_img(self):
        return self.img[:, :self.__xsplit()]

    # horizontal split point
    def __xsplit(self):
        return int(self.img.shape[1] * 0.48)

    # unit and value are in right half
    def __unitvalue_img(self):
        return self.img[:, self.__xsplit():]

    # value is first segment of right half
    def value_img(self):
        segments = self.__segments()[0:(self.__unit_idx())]
        x,y,w,h = self.__bb_of_rects(segments)
        return self.__unitvalue_img()[y:(y+h), x:(x+w)]

    # unit is combined rest segments of right half
    def unit_img(self):
        segments = self.__segments()[(self.__unit_idx()):]
        x,y,w,h = self.__bb_of_rects(segments)
        return self.__unitvalue_img()[y:(y+h), x:(x+w)]

    def __segments(self):
        if self.__segments_cached is None:
            img = self.__unitvalue_img()
            # merge characters into word blobs
            ret, img2 = cv2.threshold(cv2.blur(img, (4,4)), 240, 255, cv2.THRESH_BINARY)
            # OR: img2 = cv2.erode(img, None)
            # segment
            contours, hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            boxes = map(cv2.boundingRect, contours)
            boxes = filter(lambda (x,y,w,h): w < img.shape[1]*0.8 or h < img.shape[0]*0.8, boxes) # filter out too large ones
            boxes.sort(key=lambda (x,y,w,h): x) # sort by x-position
            self.__segments_cached = boxes
        return self.__segments_cached

    # index of first unit segment
    def __unit_idx(self):
        segments = self.__segments()
        # easy if there's just two (and avoid a corner-case)
        if len(segments) == 2:
            return 1
        # join digits
        for i, (x,y,w,h) in enumerate(segments):
            if w > self.img.shape[1]/16: return i
        # error!
        raise "Last segment expected to be a word"

    # total bounding box of rectangless
    def __bb_of_rects(self, rects):
        x0 = min(map(lambda (x,y,w,h): x, rects))
        y0 = min(map(lambda (x,y,w,h): y, rects))
        x1 = max(map(lambda (x,y,w,h): x + w, rects))
        y1 = max(map(lambda (x,y,w,h): y + h, rects))
        return (x0, y0, x1-x0, y1-y0)
