import numpy
import cv2

# Image with a number containing one or more digits
class Number:
    def __init__(self, img):
        self.img = img
        self.__segments_cached = None

    def digit_imgs(self):
        return map(lambda (x,y,w,h): self.img[y:(y+h), x:(x+w)], self.__segments())

    def __segments(self):
        if self.__segments_cached is None:
            img = 255 - self.img # contour finding works on inverted image
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = map(cv2.boundingRect, contours)
            boxes.sort(key=lambda (x,y,w,h): x) # sort by x-position
            self.__segments_cached = boxes
        return self.__segments_cached

