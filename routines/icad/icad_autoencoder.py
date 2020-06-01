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
            print("Load the pretrained autoencoder model...")
            with open('./nn_model/autoencoder/autoencoder_architecture.json','r') as f:
                self.ae_model = model_from_json(f.read())
            self.ae_model.load_weights('./nn_model/autoencoder/autoencoder_weights.h5')
        except:
            print("Cannot find the pretrained model, please train it first...")
        reconstructed_iamges = self.ae_model.predict(calibrationData)
        self.calibration_NC = (np.square(reconstructed_iamges.reshape(len(calibrationData), -1) - self.calibrationData.reshape(len(calibrationData), -1))).mean(axis=1)

        self.calibration_NC.sort()
        print(self.calibration_NC)

    def __call__(self, image):
        image = np.expand_dims(image, axis=0)
        reconstructed_iamge = self.ae_model.predict(image)
        mse = (np.square(reconstructed_iamge.reshape(1, -1) - image.reshape(1, -1))).mean(axis=1)
        print(mse)
        p = (100 - stats.percentileofscore(self.calibration_NC, mse))/float(100)
        return p