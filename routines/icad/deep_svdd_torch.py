#!/usr/bin/python3
from routines.icad.network import Autoencoder, SVDDNet
import torch.optim as optim
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

nu = 0.1

class deepSVDD():
    def __init__(self, X_train, soft_boundary=False):
        print("Initialize the SVDD...")
        self.soft_boundary = soft_boundary
        self.inputs = np.rollaxis(X_train, 3, 1)
        self.targets = np.random.randn(len(self.inputs), 1)
        self.dataset = MyDataset(self.inputs, self.targets)
        self.nu = nu
        self.R = 0.0
        self.c = None

    def ae_train(self, ae_net):
        ae_net = ae_net.to('cuda')
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(ae_net.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250], gamma=0.1)
        ae_net.train()
        for epoch in range(350):
            print('LR is: {}'.format(float(scheduler.get_lr()[0])))
            if epoch in [250]:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to('cuda')
                optimizer.zero_grad()

                outputs = ae_net(inputs)
                scores = torch.sum((outputs-inputs)**2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()
                
                loss_epoch += loss.item()
                n_batches += 1
                #print('Batch idx {}, data shape{}'.format(batch_idx, data.shape))
            
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1, 350, epoch_train_time, loss_epoch/n_batches))

        return ae_net 

    def train(self, net):
        R = torch.tensor(self.R, device='cuda')
        c = torch.tensor(self.c, device='cuda') if self.c is not None else None

        net = net.to('cuda')
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

        if c in None:
            c = self.init_center_c(dataloader, net)
            print('Center c initialized.')
        
        for epoch in range(150):
            print('LR is: {}'.format(float(scheduler.get_lr()[0])))
            if epoch in [50]:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to('cuda')
                outputs = net(inputs)
                dist = torch.sum((outputs - c)**2, dim=1)
                if self.soft_boundary:
                    scores = dist - R ** 2
                    loss = R ** 2 + (1/self.nu) * torch.mean(torch.max(torch.zeros_like(scores),scores))
                else:
                    loss = torch.mean(dist)
            
                loss = backward()
                optimizer.step()

                if (self.soft_boundary) and (epoch>=10):
                    R = torch.tensor(get_radius(dist, self.nu), device='cuda')
            
                loss_epoch += loss.item()
                n_batches += 1
        
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1, 150, epoch_train_time, loss_epoch/n_batches))

        print(R, c)
        return net, R, c
    
    def init_center_c(self,dataloader, net, eps=0.1):
        n_sample = 0
        c = torch.zeros(net.rep_dim, device='cuda')

        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to('cuda')
                outputs = net(inputs)
                n_sample += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        
        c /= n_sample

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


    def fit(self):
        # autoencoder pretrain
        self.ae_net = Autoencoder()
        try:
            self.ae_net.load_state_dict(torch.load("./pytorch_model/ae.pt"))
            self.ae_net.eval()
        except:
            self.ae_net = self.ae_train(self.ae_net)
            torch.save(self.ae_net.state_dict(), "./pytorch_model/ae.pt")
        self.net = SVDDNet() 
        self.init_network_weights_from_pretraining()

        self.net, R, c = self.train(self.net)
        R = R.cpu().data.numpy()
        c = c.cpu().data.numpy()
        torch.save(self.net.state_dict(), "./pytorch_model/deepSVDD.pt")
        return 0, 0, 0
    
    def init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.net.load_state_dict(net_dict)

    def save_model(self, path):
        pass


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self,index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)