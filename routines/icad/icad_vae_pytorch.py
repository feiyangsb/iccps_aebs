#!/usr/bin/python3
import numpy as np
from scipy import stats
import torch
from routines.icad.network import VAE

class ICAD_VAE():
    def __init__(self,trainingData, calibrationData):
        self.trainingData = trainingData
        self.calibrationData = calibrationData

        try:
            print("Load the pretrained vae model...")
            self.net = VAE()
            self.net = self.net.to('cuda')
            self.net.load_state_dict(torch.load("./pytorch_model/vae.pt"))
            self.net.eval()
        except:
            print("Cannot find the pretrained model, please train it first...")

        inputs = np.rollaxis(self.calibrationData,3,1)
        dists = []
        for i in range(len(inputs)):
            input_torch = torch.from_numpy(np.expand_dims(inputs[i], axis=0)).float()
            input_torch = input_torch.to('cuda')
            output, _, _ = self.net(input_torch)
            rep = output.cpu().data.numpy()
            reconstrution_error = (np.square(rep.reshape(1, -1) - inputs[i].reshape(1, -1))).mean(axis=1)
            #print(reconstrution_error.shape)
            dists.append(reconstrution_error)
        self.calibration_NC = np.array(dists)
        self.calibration_NC.sort()
        np.save("./calibration_vae.npy", self.calibration_NC)
        print(self.calibration_NC)
    
    def __call__(self, image):
        image = np.expand_dims(image, axis=0)
        image = np.rollaxis(image, 3, 1)
        input_torch = torch.from_numpy(image).float()
        input_torch = input_torch.to('cuda')
        output, _, _ = self.net(input_torch)
        rep = output.cpu().data.numpy()
        reconstrution_error = (np.square(rep.reshape(1, -1) - image.reshape(1, -1))).mean(axis=1)
        p = (100 - stats.percentileofscore(self.calibration_NC, reconstrution_error))/float(100)
        return p