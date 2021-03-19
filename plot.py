import math
import torch
import torchvision

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# Hyperparameters
from hypers import *

# Visualization
def save_img(filename,img):
    npimg = img.numpy()
    plt.imsave(filename, np.transpose(npimg, (1,2,0)))

# Image reconstruction    
def plot_reconstruction(filename, model, data_loader, n=24):
    x,_ = next(iter(data_loader))
    x = x[:n,:,:,:].to(device)
    with torch.no_grad():
        out, _, _= model(x.view(-1, image_size)) 
    x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
    out_grid = torchvision.utils.make_grid(x_concat).cpu().data
    save_img(filename, out_grid)

# Image generation  
def plot_generation(filename, model, n=24):
    print("Generation:")
    with torch.no_grad():        
        out = model.sample(n)
        out = out.view(-1, 1, 28, 28)

    out_grid = torchvision.utils.make_grid(out).cpu()
    save_img(filename, out_grid)

# Function draws reconstruction losses.
# Saves graph to a file.
def plot_reconst_losses(rlosses, filename):
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot()

    l = len(rlosses)
    if l < 20:
        ax.set_xticks(list(range(20)))
    else:
        step = l // 10
        ax.set_xticks(list(range(0, l + step, step)))

    ax.set_xlabel("Iteration (batch number)")
    ax.set_ylabel("Reconstruction loss")
    
    ax.plot(rlosses)
    plt.savefig(filename)
    
# Function draws train/val losses
# Saves graph to a file.
def plot_tv_losses(tlosses, vlosses, filename):
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot()

    l = len(tlosses)
    if l < 20:
        ax.set_xticks(list(range(20)))
    else:
        step = l // 10
        ax.set_xticks(list(range(0, l + step, step)))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss on test / val sets")
    
    ax.plot(tlosses, label="train total loss")
    ax.plot(vlosses, label="valid total loss")

    ax.legend()
    plt.savefig(filename)
