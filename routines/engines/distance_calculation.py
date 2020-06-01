import sys
import os
import cv2
try:
    sys.path.append(os.environ["CARLA_PYTHON"])
    import carla
    from carla import Image
except:
    raise Exception('No CARLA module found.')
import numpy as np
from keras.models import model_from_json
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

class DistanceCalculation():
    def __init__(self, ego_vehicle, leading_vehicle, perception=None):
        self.ego_vehicle = ego_vehicle
        self.leading_vehicle = leading_vehicle
        self.perception = perception
        if perception is not None:
            with open(os.path.join(perception, "perception_architecture.json"), 'r') as f:
                self.model = model_from_json(f.read())
            self.model.load_weights(os.path.join(perception, 'perception_weights.h5'))


    def getTrueDistance(self):
        distance = self.leading_vehicle.get_location().y - self.ego_vehicle.get_location().y \
                - self.ego_vehicle.bounding_box.extent.x - self.leading_vehicle.bounding_box.extent.x
        return distance 
    
    def getRegressionDistance(self, image):
        if self.perception is not None:
            img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img = np.reshape(img, (image.height, image.width, 4))
            img = cv2.resize(img, (224,224))
            img = img[:, :, :3]/255.
            #img = img[:, :, ::-1]/255.
            img = np.expand_dims(img,axis=0)
            distance = self.model.predict(img)*120.0
            return float(distance[0][0])
        return None
    
    def getAttackDistance(self, image, true_distance):
        if self.perception is not None:
            img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img = np.reshape(img, (image.height, image.width, 4))
            img = cv2.resize(img, (224,224))
            img = img[:, :, :3]/255.
            #img = img[:, :, ::-1]/255.
            img = np.expand_dims(img,axis=0)
            adv_x, adv_y = self.generate_adv(img, true_distance/120.0, 0.02, lr=1.0, steps=200)
            return adv_x, adv_y[0][0]*120.0
        return None
        
    def generate_adv(self, image, label, bound, lr=1e-1, steps=100):
    
        img = tf.Variable(tf.zeros((1, 224, 224, 3)))


        prediction = self.model(img)

        x = tf.placeholder(tf.float32, (1, 224, 224, 3))

        x_hat = img  # our trainable adversarial input

        assign_op = tf.assign(x_hat, x)

        learning_rate = tf.placeholder(tf.float32, ())
        y_hat = tf.placeholder(tf.float32, (1, 1))

        loss = tf.losses.mean_squared_error(labels=y_hat, predictions=prediction)
        optim_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, var_list=[x_hat])

        epsilon = tf.placeholder(tf.float32, ())

        below = x - epsilon
        above = x + epsilon
        projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
        with tf.control_dependencies([projected]):
            project_step = tf.assign(x_hat, projected)

        target = [[1]]  # maximum dist
        # initialization step
        sess.run(assign_op, feed_dict={x: image})

        # projected gradient descent
        for i in range(steps):
            # gradient descent step
            _, loss_value = sess.run(
                [optim_step, loss],
                feed_dict={learning_rate: lr, y_hat: target})
            # pred_value = sess.run(prediction, feed_dict={x: image})
            # print(prediction.eval())
            # project step

            sess.run(project_step, feed_dict={x: image, epsilon: bound})
            #if (i + 1) % 10 == 0:
            #    print('step %d, loss=%g' % (i + 1, loss_value))

        # print(x_hat.eval())

        adv_x = x_hat.eval()  # retrieve the adversarial example
        adv_y = self.model(x_hat).eval()
        return adv_x, adv_y