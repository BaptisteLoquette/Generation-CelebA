import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.lin1 = nn.Linear(4096, 2048)
        self.lin2 = nn.Linear(2048,1024)
        self.logvar = nn.Linear(1024,100)
        self.mu = nn.Linear(1024,100)
        
        self.lin3 = nn.Linear(100, 1024)
        self.lin4 = nn.Linear(1024, 2048)
        self.lin5 = nn.Linear(2048,4096)
        
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, X):
        """
        Inputs :
            -   X   :   The Ground Truth Image Input (batch_size, 64, 64)

        Returns :
            -   mean    :   The Encoded Image Mean Vector
            -   logstd  :   The Encoded Image Standard Deviation Vector (Log for computational stability)
        """
        x = self.lrelu(self.lin1(X.view(-1, 4096)))
        x = self.lrelu(self.lin2(x))
        mean = self.mu(x)
        logstd = self.logvar(x)

        return mean, logstd
    
    def z(self, mean, logstd):
        """
        Inputs :
            -   mean    :   The Encoded Image Mean Vector
            -   logstd  :   The Encoded Image Standard Deviation Vector (Log for computational stability)
        Returns :
            -   z       :   The Reparametrized Latent Vector
        """
        std     =   torch.exp(0.5*logstd)
        noise   =   torch.randn_like(std)

        z   =   mean + std*noise
        
        return z
    
    def decoder(self, Z):
        """
        Inputs :
            -   Z   :   The Latent Vector   (batch_size, 100)
            -   I   :   The Decoded Images Arrays (batch_size, 4096)
        """
        x = self.lrelu(self.lin3(Z))
        x = self.lrelu(self.lin4(x))
        I = self.lin5(x)
        
        return I
    
    def forward(self, X):

        mean, logstd    =   self.encoder(X)
        codages         =   self.z(mean, logstd)
        x               =   self.decoder(codages)
        
        return x, mean, logstd