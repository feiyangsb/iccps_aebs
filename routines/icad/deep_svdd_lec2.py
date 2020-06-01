#!/usr/bin/python3

from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Dense
from keras.models import Model, model_from_json
import numpy as np
from keras.callbacks import Callback, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import os

nu = 0.2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, X_train):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.radius = radius
        self.model = model
        self.inputs = X_train
        self.cvar = cvar
        self.y_reps = np.zeros((len(X_train), 4))

    def on_epoch_end(self, batch, logs={}):

        reps = self.model.predict(self.inputs)
        self.y_reps = reps
        center = self.cvar
        dist = np.sum((reps - self.cvar) ** 2, axis=1)
        scores = dist
        val = np.sort(scores)
        R_new = np.percentile(val, nu * 100)  # qth quantile of the radius.
        # print("[INFO:] Center (c)  Used.", center)
        # print("[INFO:] Updated Radius (R) .", R_updated)
        self.radius = R_new
        print("[INFO:] \n Updated Radius Value...", R_new)
        # print("[INFO:] \n Updated Rreps value..", self.y_reps)
        return self.radius

class deepSVDD():
    def __init__(self,X_train, soft_boundary=False):
        print("Initialize the SVDD...")
        self.inputs = X_train
        self.Rvar = 1.0
        self.cvar = 0.0
        self.soft_boundary = soft_boundary
    
    def create_model(self):
        input_img = Input(shape=(100, 512, 3))
        x = Conv2D(32, (5, 5),  use_bias=False, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(64, (5, 5), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(128, (5, 5), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)


        x = Conv2D(256, (5, 5), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(2048)(x)
        x = LeakyReLU(0.1)(x)

        x = Dense(256, use_bias=False)(x)
        x = LeakyReLU(0.1)(x)

        model = Model(input_img, x)
        model.summary()
        return model

    def initialize_c_with_mean(self, inputs, model):
        reps = model.predict(inputs)
        
        eps = 0.1
        c = np.mean(reps, axis=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps

        self.cvar = c

        dist = np.sum((reps - c) ** 2, axis=1)
        val = np.sort(dist)
        self.Rvar = np.percentile(val, nu * 100)

        print("Radius initialized.", self.Rvar)

    def initialize_c_with_mean_with_generator(self, inputs, model):
        for (input_batch, targets_batch) in inputs:
            reps = model.predict(input_batch)
        
            eps = 0.1
            c = np.mean(reps, axis=0)
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c >= 0)] = eps

            self.cvar = c

            dist = np.sum((reps - c) ** 2, axis=1)
            val = np.sort(dist)
            self.Rvar = np.percentile(val, nu * 100)

            print("Radius initialized.", self.Rvar)
        # Custom loss SVDD_loss ball interpretation
    def custom_ocnn_hypershere_loss(self):

        center = self.cvar

        # val = np.ones(Cfg.mnist_rep_dim) * 0.5
        # center = K.variable(value=val)


        # define custom_obj_ball
        def custom_obj_ball(y_true, y_pred):
            # compute the distance from center of the circle to the

            dist = (K.sum(K.square(y_pred - center), axis=1))
            avg_distance_to_c = K.mean(dist)

            return (avg_distance_to_c)

        return custom_obj_ball
    
    def custom_ocnn_hyperplane_loss(self, center, r):

        def custom_hinge(y_true, y_pred):
            # term1 = 0.5 * tf.reduce_sum(w ** 2)
            # term2 = 0.5 * tf.reduce_sum(V ** 2)

            term3 =   K.square(r) + K.sum( K.maximum(0.0,    K.square(y_pred -center) - K.square(r)  ) , axis=1 )
            # term3 = K.square(r) + K.sum(K.maximum(0.0, K.square(r) - K.square(y_pred - center)), axis=1)
            term3 = 1 / nu * K.mean(term3)

            loss = term3

            return (loss)

        return custom_hinge

    def fit(self):
        self.model_svdd = self.create_model()
        self.initialize_c_with_mean(self.inputs, self.model_svdd)

        out_batch = Adjust_svdd_Radius(self.model_svdd, self.cvar, self.Rvar, self.inputs)

        def lr_scheduler(epoch):
            lr = 1e-4
            if epoch > 50:
                lr = 1e-5
                if(epoch== 51):
                    print('lr: rate adjusted for fine tuning %f' % lr)

            # print('lr: %f' % lr)
            return lr

        scheduler = LearningRateScheduler(lr_scheduler)
        opt = Adam(lr=1e-4)
        callbacks = [out_batch, scheduler]
        if self.soft_boundary:
            self.model_svdd.compile(loss=self.custom_ocnn_hyperplane_loss(self.cvar, out_batch.radius.astype(np.float32)), optimizer=opt)
        else:
            self.model_svdd.compile(loss=self.custom_ocnn_hypershere_loss(), optimizer=opt)
        y_reps = out_batch.y_reps
        self.Rvar = out_batch.radius

        self.model_svdd.fit(self.inputs, y_reps, shuffle=True, batch_size=64, epochs=150, validation_split=0.01, verbose=0, callbacks=callbacks)
        self.Rvar = out_batch.radius
        self.cvar = out_batch.cvar
        print(self.cvar, self.Rvar)
        return self.model_svdd, self.cvar, self.Rvar
    
    def fit_generator(self):
        self.model_svdd = self.create_model()
        self.initialize_c_with_mean(self.inputs, self.model_svdd)

        #out_batch = Adjust_svdd_Radius(self.model_svdd, self.cvar, self.Rvar, self.inputs)

        def lr_scheduler(epoch):
            lr = 1e-4
            if epoch > 50:
                lr = 1e-5
                if(epoch== 51):
                    print('lr: rate adjusted for fine tuning %f' % lr)

            # print('lr: %f' % lr)
            return lr

        scheduler = LearningRateScheduler(lr_scheduler)
        opt = Adam(lr=1e-4)
        callbacks = [out_batch, scheduler]
        if self.soft_boundary:
            self.model_svdd.compile(loss=self.custom_ocnn_hyperplane_loss(self.cvar, out_batch.radius.astype(np.float32)), optimizer=opt)
        else:
            self.model_svdd.compile(loss=self.custom_ocnn_hypershere_loss(), optimizer=opt)
        y_reps = out_batch.y_reps
        self.Rvar = out_batch.radius

        self.model_svdd.fit_generator(self.inputs, y_reps, shuffle=True, batch_size=64, epochs=150, validation_split=0.01, verbose=0, callbacks=callbacks)
        self.Rvar = out_batch.radius
        self.cvar = out_batch.cvar
        print(self.cvar, self.Rvar)
        return self.model_svdd, self.cvar, self.Rvar


    def save_model(self, path):
        self.model_svdd.save_weights(os.path.join(path, "svdd_weights.h5"))
        with open(os.path.join(path, 'svdd_architecture.json'), 'w') as f:
            f.write(self.model_svdd.to_json())
        np.save(os.path.join(path, 'svdd_center'), self.cvar)
