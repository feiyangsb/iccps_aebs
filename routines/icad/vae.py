from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D, Activation, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.models import Model, model_from_json
from keras.losses import mse
import numpy as np
from keras import backend as K
import tensorflow as tf
import os


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE():
    def __init__(self,X_train):
        print("Initialize the autoencoder...")
        self.inputs = X_train
        self.latent_dim = 1024

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
        
        x = Conv2D(16, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(2048)(x)
        x = LeakyReLU(0.1)(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log')(x)
        
        z = Lambda(sampling, output_shape=(self.latent_dim, ), name='z')([z_mean, z_log_var]) 

        encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(2048)(latent_inputs)
        x = LeakyReLU(0.1)(x)

        x = Dense(3136)(x)
        x = LeakyReLU(0.1)(x)

        x = Reshape((14, 14, 16))(x)

        x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2,2))(x)

        x = Conv2D(3, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)
        
        decoder = Model(latent_inputs, decoded)
        decoder.summary()
        
        outputs = decoder(encoder(input_img)[2])
        vae = Model(input_img,outputs)
        vae.summary()

        reconstruction_loss = mse(K.flatten(input_img), K.flatten(outputs))
        reconstruction_loss *= 224*224*3
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        return vae


    def fit(self):
        self.model = self.create_model()
        self.model.fit(self.inputs, epochs=100, batch_size=16, shuffle=True)

        return self.model
    
    def save_model(self, path):
        self.model.save_weights(os.path.join(path, "vae_weights.h5"))
        with open(os.path.join(path, 'vae_architecture.json'),'w') as f:
            f.write(self.model.to_json())
