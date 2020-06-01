import sys
import os
import cv2
import torch
from routines.network import VAEPerceptionNet
from scipy import stats
try:
    sys.path.append(os.environ["CARLA_PYTHON"])
    import carla
    from carla import Image
except:
    raise Exception('No CARLA module found.')
import numpy as np


class DistanceCalculation():
    def __init__(self, ego_vehicle, leading_vehicle, perception=None):
        self.ego_vehicle = ego_vehicle
        self.leading_vehicle = leading_vehicle
        self.perception = perception
        if perception is not None:
            model_path = "./nn_model/vae_regression/perception.pt"
            self.model = VAEPerceptionNet()
            self.model = self.model.to('cuda')
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.calibration_NC = np.load("./nn_model/vae_regression/calibration.npy")


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
            img = np.rollaxis(img, 3,1)
            input_img = torch.from_numpy(img)
            image = input_img.to(device='cuda', dtype=torch.float)
            p_list = []
            for i in range(20):
                with torch.no_grad():
                    output, _, _, _, distance, _ = self.model(image)
                    rep = output.cpu().data.numpy()
                    reconstruction_error = (np.square(rep.reshape(1, -1) - img.reshape(1, -1))).mean(axis=1)
                    p = (100 - stats.percentileofscore(self.calibration_NC, reconstruction_error))/float(100)
                    p_list.append(p)

            distance = distance.cpu().numpy()*120.0
            return float(distance[0][0]), p_list
        return None
    
    def getAttackDistance(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        img = cv2.resize(img, (224,224))
        img = img[:, :, :3]/255.
        #img = img[:, :, ::-1]/255.
        img = np.expand_dims(img,axis=0)
        test_image = np.rollaxis(img, 3, 1)
        input_torch = torch.from_numpy(test_image).float()
        input_torch = input_torch.to('cuda')
        loss_fcn = torch.nn.MSELoss()
        input_torch.requires_grad = True
        alpha = 0.3
        epsilon = 0.02
        distance_target_numpy = np.asarray([[1.0]])
        distance_target = torch.from_numpy(distance_target_numpy).float().to('cuda')
        self.model.zero_grad()
        _,_,_,_,r_mu,_ = self.model(input_torch)
        loss = loss_fcn(r_mu, distance_target)
        loss.backward()
        perturbation = alpha * torch.sign(input_torch.grad.data)
        perturbation = torch.clamp((input_torch.data + perturbation) - input_torch, min=-epsilon, max=epsilon)
        input_torch.data = input_torch - perturbation
        p_list = []
        for i in range(20):
            with torch.no_grad():
                output,_,_,_,r_mu,_ = self.model(input_torch)
                rep = output.cpu().data.numpy()
                reconstruction_error = (np.square(rep.reshape(1, -1) - img.reshape(1, -1))).mean(axis=1)
                p = (100 - stats.percentileofscore(self.calibration_NC, reconstruction_error))/float(100)
                p_list.append(p)
        distance = r_mu.cpu().numpy()*120.0
        return distance[0][0], p_list