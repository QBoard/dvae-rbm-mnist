import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import math
import numpy as np
import pickle
import time

import dvae
import evalm
import plot

# Hyperparameters
from hypers import * 

print("Device = ", device)
print("Mini sets =", mini_sets)
print("z_dim = ", z_dim)
print("numh =", numh)
print("learning rate =", learning_rate)

# MNIST dataset
dataset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
# Test set
test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())

print("Dataset size=", len(dataset))
#seed_num = 87960471325021
seed_num = 57
print("seed num = ", seed_num)
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(seed_num))


# Data loaders
# Train
data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=batch_size, 
                                          shuffle=True)
# Test
test_loader = torch.utils.data.DataLoader(dataset = test_set,
                                          batch_size=batch_size,
                                          shuffle=True)

if not mini_sets:
    # Validation
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=batch_size, 
                                          shuffle=True)

    train_total_loader = data_loader
    valid_total_loader = val_loader
    test_total_loader = test_loader

else:
    _, mini_train_set = torch.utils.data.random_split(train_set, [49000, 1000], generator=torch.Generator().manual_seed(seed_num))
    mini_data_loader = torch.utils.data.DataLoader(dataset=mini_train_set,
                                          batch_size=batch_size, 
                                          shuffle=True)

    _, mini_val_set = torch.utils.data.random_split(val_set, [9000, 1000], generator=torch.Generator().manual_seed(seed_num))
    mini_val_loader = torch.utils.data.DataLoader(dataset=mini_val_set,
                                          batch_size=batch_size, 
                                          shuffle=True)
    
    _, mini_test_set = torch.utils.data.random_split(test_set, [9000, 1000], generator=torch.Generator().manual_seed(seed_num))
    mini_test_loader = torch.utils.data.DataLoader(dataset = mini_test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    train_total_loader = mini_data_loader
    valid_total_loader = mini_val_loader
    test_total_loader = mini_test_loader

print("len train_total_loader =", len(train_total_loader))
print("len valid_total_loader =", len(valid_total_loader))
print("len test_total_loader =", len(test_total_loader))
    
# Change to results dir
if not os.path.isdir(results_dir):
    print("Directory ", results_dir, " does not exist.")
    os.mkdir(results_dir)
    print("Created directory ", results_dir)
os.chdir(results_dir)
print("Results will be saved to:", os.getcwd())

smplr = Sampler(z_dim)
model = dvae.DVAE(smplr, image_size, h_dim, z_dim, numh, beta).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epoch = 0

if (len(sys.argv) > 1):
    print("Loading checkpoint...")
    print(sys.argv[1])
    checkpoint = torch.load(sys.argv[1])
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Done!")

rec_losses = []    

train_total_losses = []
valid_total_losses = []

# Start training
while epoch < num_epochs:
    model.train()
    t1 = time.time()
    for i, (x, _) in enumerate(data_loader):
        x_size = x.size(0)
        
        # Forward pass
        x = x.to(device).view(-1, image_size)
        x_reconst, ent, p_ph = model(x)

        # Compute reconstruction loss
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
        
        # Calculation of KL divergence without term containing the partition function
        kl_div = -ent + p_ph
        
        if (i+1) % 10 == 0:
            with torch.no_grad():
                logZ = evalm.calc_logZ(model)
                
        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        model.updateBMgradients(num_samples_pb, x_size) # negative phase
        optimizer.step()
        
        rec_losses.append(reconst_loss.item()/x_size)
        
        if (i+1) % 10 == 0:
            rl = reconst_loss.item()/x_size
            kld = kl_div.item()/x_size + logZ
            tl = rl + kld
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.5f}, KL Div: {:.5f}, Total Loss: {:.5f}" 
                   .format(epoch+1, num_epochs, i+1, len(data_loader), rl, kld, tl)) 
    
    print("Epoch =", epoch+1)
    t2 = time.time()
    print("Training epoch time = ", t2-t1)
    
    # Model evaluation
    train_total_loss = evalm.evaluate_model(model, train_total_loader)
    valid_total_loss = evalm.evaluate_model(model, valid_total_loader)
    print("train total loss = ", train_total_loss)
    print("valid total loss = ", valid_total_loss)
    
    train_total_losses.append(train_total_loss)
    valid_total_losses.append(valid_total_loss)
    t3 = time.time()
    print("Total epoch time = ", t3-t1)
    
    epoch += 1
    print("Saving checkpoint...")
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, "checkpoint.pt")
    print("Done!")    
    
test_total_loss = evalm.evaluate_model(model, test_total_loader)
print("Total loss on test data:", test_total_loss)

# Saving reconstruction losses
with open('reconst_losses', 'wb') as f:
    pickle.dump(rec_losses, f)

# Plot reconstruction losses
plot.plot_reconst_losses(rec_losses, 'reconst_losses.png')

# Saving train total losses
with open('train_total_losses', 'wb') as f:
    pickle.dump(train_total_losses, f)

# Saving valid total losses
with open('valid_total_losses', 'wb') as f:
    pickle.dump(valid_total_losses, f)

# Plot train and valid total losses
plot.plot_tv_losses(train_total_losses, valid_total_losses, 'tv_total_losses.png')
    
plot.plot_reconstruction('rec1.png', model, test_loader)
plot.plot_reconstruction('rec2.png', model, test_loader)
plot.plot_reconstruction('rec3.png', model, test_loader)
plot.plot_reconstruction('rec4.png', model, test_loader)
plot.plot_reconstruction('rec5.png', model, test_loader)

plot.plot_generation('gen1.png', model)
plot.plot_generation('gen2.png', model)
plot.plot_generation('gen3.png', model)
plot.plot_generation('gen4.png', model)
plot.plot_generation('gen5.png', model)
