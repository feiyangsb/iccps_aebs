from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D, Activation
from keras.models import Model, model_from_json
import numpy as np
from keras import backend as K
import tensorflow as tf
import os


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class Autoencoder():
    def __init__(self,X_train):
        print("Initialize the autoencoder...")
        self.inputs = X_train
    
    def create_model(self):
        input_img = Input(shape=(self.inputs.shape[1], self.inputs.shape[2], self.inputs.shape[3]))
        x = Conv2D(128, (3, 3),  use_bias=False, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        encoded = LeakyReLU(0.1)(x)

        x = Conv2D(32, (3, 3), use_bias=False, padding='same')(encoded)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(3, (3, 3), use_bias=False, padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)
        
        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        return autoencoder


    def fit(self):
        self.model = self.create_model()
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(self.inputs, self.inputs, epochs=100, batch_size=16, shuffle=True)

        return self.model
    
    def save_model(self, path):
        self.model.save_weights(os.path.join(path, "autoencoder_weights.h5"))
        with open(os.path.join(path, 'autoencoder_architecture.json'),'w') as f:
            f.write(self.model.to_json())