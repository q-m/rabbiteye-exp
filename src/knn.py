import numpy
import cv2

class KNN:
    def __init__(self):
        self.knn = cv2.KNearest()

    def train(self, imgs, labels):
        # convert string labels to index
        self.labels = list(set(labels))
        self.__train_labels = numpy.array(map(self.labels.index, labels))[:, numpy.newaxis]
        # put images into array
        self.__shape = (
                max(map(lambda i: i.shape[0], imgs)),
                max(map(lambda i: i.shape[1], imgs))
        )
        self.__train_imgs = numpy.vstack(map(self.__prepare, imgs))
        # train
        self.knn.train(self.__train_imgs, self.__train_labels)

    def find(self, img, k=3):
        r = self.find_nearest(img, k)[0]
        if r is not None: return self.labels[r]

    def find_nearest(self, img, k=3):
        return self.knn.find_nearest(self.__prepare(img), k=k)

    def save(self, filename):
        numpy.savez(filename,
                train_imgs=self.__train_imgs.astype(numpy.uint8),
                train_labels=self.__train_labels,
                labels=self.labels,
                shape=self.__shape)

    def load(self, filename):
        with numpy.load(filename) as data:
            self.labels = data['labels']
            self.__shape = data['shape']
            self.__train_labels = data['train_labels']
            self.__train_imgs = data['train_imgs'].astype(numpy.float32)
            self.knn.train(self.__train_imgs, self.__train_labels)

    def __prepare(self, img):
        padded = numpy.zeros(self.__shape)
        padded[0:(img.shape[0]),0:(img.shape[1])] = img
        return padded.reshape(-1, self.__shape[0]*self.__shape[1]).astype(numpy.float32)
