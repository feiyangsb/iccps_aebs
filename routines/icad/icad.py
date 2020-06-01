#!/usr/bin/python3
from keras.models import model_from_json
import numpy as np
from scipy import stats
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class ICAD():
    def __init__(self,trainingData, calibrationData):
        self.trainingData = trainingData
        self.calibrationData = calibrationData

        try:
            print("Load the pretrained svdd model...")
            with open('./nn_model/deep_svdd/svdd_architecture.json','r') as f:
                self.svdd_model = model_from_json(f.read())
            self.svdd_model.load_weights('./nn_model/deep_svdd/svdd_weights.h5')
            self.center = np.load('./nn_model/deep_svdd/svdd_center.npy')
        except:
            print("Cannot find the pretrained model, please train it first...")

        reps = self.svdd_model.predict(self.calibrationData)
        dists = np.sum((reps-self.center) ** 2, axis=1)
        self.calibration_NC = dists
        self.calibration_NC.sort()
        print(self.calibration_NC.shape)
    
    def __call__(self, image):
        image = np.expand_dims(image, axis=0)
        rep = self.svdd_model.predict(image)
        dist = np.sum((rep-self.center)**2, axis=1)
        print(dist)
        p = (100 - stats.percentileofscore(self.calibration_NC, dist))/float(100)
        return p