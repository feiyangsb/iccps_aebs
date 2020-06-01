#!/usr/bin/python
import numpy as np
import os
from routines.perception.data_augmentation import Augumentation
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, dataPath='./parser_data_1'):
        self.data_path = dataPath
    
    def load(self):
        images = np.load(os.path.join(self.data_path, 'X.npy'))
        labels = np.load(os.path.join(self.data_path, 'Y.npy'))
        #images = np.concatenate((images, np.load("./data/MidRainSunset_training/X.npy")))
        #labels = np.concatenate((labels, np.load("./data/MidRainSunset_training/Y.npy")))
        print("There are totally {} images using for training.".format(len(images)))
        X_train, X_calibration, Y_train, Y_calibration = train_test_split(images, labels, test_size=0.2, random_state=42)
        print("# of training images: {}, # of calibration images: {}".format(len(X_train), len(X_calibration)))
        return (X_train, Y_train), (X_calibration, Y_calibration) 
