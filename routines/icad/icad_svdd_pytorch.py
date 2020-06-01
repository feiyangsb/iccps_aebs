#!/usr/bin/python3
import numpy as np
from scipy import stats
import torch
from routines.icad.network import SVDDNet

class ICAD_SVDD():
    def __init__(self,trainingData, calibrationData):
        self.trainingData = trainingData
        self.calibrationData = calibrationData

        try:
            print("Load the pretrained svdd model...")
            self.net = SVDDNet()
            self.net = self.net.to('cuda')
            self.net.load_state_dict(torch.load("./pytorch_model/deepSVDD_vae.pt"))
            self.net.eval()
            self.center = np.load('./pytorch_model/svdd_c_vae.npy')
        except:
            print("Cannot find the pretrained model, please train it first...")

        inputs = np.rollaxis(self.calibrationData,3,1)
        dists = []
        for i in range(len(inputs)):
            input_torch = torch.from_numpy(np.expand_dims(inputs[i], axis=0)).float()
            input_torch = input_torch.to('cuda')
            output = self.net(input_torch)
            rep = output.cpu().data.numpy()
            dist = np.sum((rep - self.center)**2, axis=1)
            dists.append(dist)
        self.calibration_NC = np.array(dists)
        self.calibration_NC.sort()
        np.save("./ncm_calibration.npy", self.calibration_NC)
    
    def __call__(self, image):
        image = np.expand_dims(image, axis=0)
        image = np.rollaxis(image, 3, 1)
        input_torch = torch.from_numpy(image).float()
        input_torch = input_torch.to('cuda')
        output = self.net(input_torch)
        rep = output.cpu().data.numpy()
        dist = np.sum((rep - self.center)**2, axis=1)
        p = (100 - stats.percentileofscore(self.calibration_NC, dist))/float(100)
        return p