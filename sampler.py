import torch
import numpy as np
import math
import time

# Hyperparameters
from hypers import *

# Gibbs Sampler class
class GSampler:
    def __init__(self, z_dim, gc_num=30):
        self.name = 'gibbs'
        self.z_dim = z_dim        
        self.gc_num = gc_num
        self.z = torch.ones(2 * z_dim, device=device) * 0.5
        self.z = torch.bernoulli(self.z) 
    
    def sample(self, W, b, num_samples):
        # Implements Gibbs sampler
        z_samples = torch.zeros(num_samples, 2 * self.z_dim).to(device)
        
        for j in range(num_samples):
            for i in range(self.gc_num):
                pv = torch.sigmoid(torch.mm(self.z[self.z_dim:].view(1,self.z_dim), -W.t()) - b[:self.z_dim])
                self.z[:self.z_dim] = torch.bernoulli(pv)
                ph = torch.sigmoid(torch.mm(self.z[:self.z_dim].view(1,self.z_dim), -W) - b[self.z_dim:])
                self.z[self.z_dim:] = torch.bernoulli(ph)
            z_samples[j] = self.z
            
        #print("z =", self.z)
            
        return z_samples

