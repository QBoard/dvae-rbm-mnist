import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

# Hyperparameters
from hypers import *

# DVAE model
class DVAE(nn.Module):
    def __init__(self, sampler, image_size=784, h_dim=500, z_dim=20, numh=1, beta=1.0):
        super(DVAE, self).__init__()
        assert numh == 1 or numh == 2
        self.sampler = sampler
        self.image_size = image_size
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.numh = numh
        self.beta = beta

        self.efc1 = nn.Linear(self.image_size, self.h_dim)
        self.bn1 = nn.BatchNorm1d(self.h_dim)
        self.efc2 = nn.Linear(self.h_dim, (3-self.numh) * self.z_dim)
        
        if numh == 2:
            self.efc3 = nn.Linear(self.image_size + self.z_dim, self.h_dim)
            self.bn2 = nn.BatchNorm1d(self.h_dim)
            self.efc4 = nn.Linear(self.h_dim, self.z_dim)
            
        self.dfc1 = nn.Linear(2 * self.z_dim, self.h_dim)        
#        self.bn3 = nn.BatchNorm1d(self.h_dim)
        self.dfc3 = nn.Linear(self.h_dim, self.image_size)

        #RBM params
        self.W = nn.Parameter(torch.randn(self.z_dim, self.z_dim ) * 0.01)
        self.b = nn.Parameter(torch.randn(2 * self.z_dim ) * 0.01)

        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                #nn.init.xavier_uniform_(p)        
                nn.init.kaiming_uniform_(p)
        
    def encode(self, x):
        # encoder
        h1 = F.relu(self.bn1(self.efc1(x)))
        q = torch.sigmoid(self.efc2(h1))
        z, zeta = self.reparameterize(q)

        if self.numh == 2:
            xx = torch.cat((x,zeta),1)
            h1 = F.relu(self.bn2(self.efc3(xx)))
            q1 = torch.sigmoid(self.efc4(h1))
            z1, zeta1 = self.reparameterize(q1)
            
            q = torch.cat((q,q1),1)
            z = torch.cat((z,z1),1)
            zeta = torch.cat((zeta,zeta1),1)
        
        return q, z, zeta
    
    def reparameterize(self, q):
        # reparametrization
        u = torch.rand(q.shape).to(device)
        z = torch.sign(u+q-1)
        z[z<0] = 0
        zeta = torch.log(torch.div(torch.max(u+q-1.0,torch.tensor(0.0,device=device)),q + eps)*(math.exp(self.beta) - 1.0)+1.0)/self.beta
        return z, zeta

    def decode(self, zeta):
        # decoder
#        h1 = F.relu(self.bn3(self.dfc1(zeta)))
        h1 = F.relu(self.dfc1(zeta))
        return torch.sigmoid(self.dfc3(h1))

    def forward(self, x):
        q, z, zeta = self.encode(x)
        x_reconst = self.decode(zeta)
        
        # entropy
        ent = q.clone()
        ent[z<0.5] = 1 - ent[z<0.5]
        ent = torch.log(ent + eps)
        ent = -torch.sum(ent)

        # part of cross-entropy corresponding to the positive phase
        p_ph = torch.einsum('bi,ij,bj->b',z[:,:self.z_dim], self.W, z[:,self.z_dim:])
        p_ph = p_ph.view(len(p_ph),1) + torch.mm(z, self.b.view(len(self.b),1))
        p_ph = torch.sum(p_ph)
        return x_reconst, ent, p_ph

    def updateBMgradients(self, num_samples_pb, batch_size):
        # Calculates negative phase for RBM
        # (the derivative of the part of cross-entropy containing the partition function)
        with torch.no_grad():
            z_samples = self.sampler.sample(self.W, self.b, num_samples_pb)
        
        dW = torch.zeros(self.z_dim, self.z_dim).to(device)
        db = torch.zeros(2 * self.z_dim).to(device)
        for i in range(num_samples_pb):
            dW += torch.mm(z_samples[i,:self.z_dim].view(1,self.z_dim).t(),z_samples[i,self.z_dim:].view(1,self.z_dim))
            db += z_samples[i]
        
        self.W.grad = self.W.grad - dW / num_samples_pb * batch_size
        self.b.grad = self.b.grad - db / num_samples_pb * batch_size
        
    def sample(self, n):
        z = self.sampler.sample(self.W, self.b, n)
        print("z = \n", z[:10])
        
        u = torch.rand(n, 2 * self.z_dim).to(device)
        zeta = z * torch.log(u * (math.exp(self.beta)-1.0)+1.0+eps) / self.beta
        out = self.decode(zeta)        
        return out
