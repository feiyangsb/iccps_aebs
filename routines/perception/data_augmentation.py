#!/usr/bin/python3

import cv2
import numpy as np

class Augumentation(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y/120.0
    
    def __call__(self, filter=False):
        images = []
        y = []
        for i in range(len(self.X)):
            if self.y[i] < 110.0/120.0 and not filter:
                image = cv2.resize(self.X[i], (224,224))
                images.append(image)
                y.append(self.y[i])
            if self.y[i] < 10.0/120.0 and filter:
                image = cv2.resize(self.X[i], (224,224))
                images.append(image)
                y.append(self.y[i])
        images = np.asarray(images)/255.0
        y = np.asarray(y)
        return images, y