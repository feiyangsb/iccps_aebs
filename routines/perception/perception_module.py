from keras.models import Sequential
from keras.layers import Conv2D, Dense,Dropout
from keras.layers import Activation, Flatten, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD

import tensorflow as tf
from keras.models import model_from_json
import os
import sys
import numpy as np
import cv2
try:
    sys.path.append(os.environ["CARLA_PYTHON"])
    from carla import Image
except:
    raise Exception('No CARLA module found.')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class DistanceTrainer(object):
    def __init__(self, X, y):
        self.X = X 
        self.y = y
        self.row = self.X.shape[1]
        self.col = self.X.shape[2]
        self.model = self.create_model()
        self.model.summary()
    
    def create_model(self, momentum=.9, weight_penalty=0.0001):
        input_shape = (self.row, self.col, 3)
        model = Sequential() 
        model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="valid" ,use_bias=False,kernel_regularizer=l2(weight_penalty),input_shape=input_shape))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(Conv2D(36, (5, 5), strides=(2, 2), padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(Conv2D(48, (5, 5), strides=(2, 2),padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding="valid",use_bias=False,kernel_regularizer=l2(weight_penalty)))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(100, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
        model.add(Dense(50, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
        model.add(Dense(10, activation='relu',kernel_initializer='he_normal',bias_regularizer=l2(weight_penalty),kernel_regularizer=l2(weight_penalty)))
        model.add(Dense(1,activation='sigmoid',kernel_initializer='he_normal')) 
        return model
    
    def fit(self):
        optim = SGD(lr=0.01, momentum=0.9, nesterov=True, clipnorm=1.)
        self.model.compile(loss='mse', optimizer=optim, metrics=['mae'])
        self.model.fit(self.X, self.y, batch_size=64, epochs=100, validation_split=0.1, shuffle=True)
    
    def save_model(self, path):
        self.model.save_weights(os.path.join(path, "perception_weights.h5"))
        with open(os.path.join(path, 'perception_architecture.json'),'w') as f:
            f.write(self.model.to_json())

class NNController(object):
    def __init__(self, path):
        with open(os.path.join(path, 'control_architecture.json'), 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(os.path.join(path, 'control_weights.h5'))
    def predict(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        img = cv2.resize(img, (400,300))
        img = img[:, :, :3]
        img = img[:, :, ::-1]/255.0
        img = np.expand_dims(img, axis=0)
        steering = self.model.predict(img)/20.0
        #print(img.shape, steering)
        return float(steering[0][0])