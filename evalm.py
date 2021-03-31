import math
import torch
import torch.nn.functional as F

# Hyperparameters
from hypers import *

# Estimation of logZ, Z - partition function
# Implements Annealed Importance Sampling (AIS)
def calc_logZ(model):
    s = model.sampler   # Sampler(model.z_dim)
    c = [x/ais_num_dists for x in range(0,ais_num_dists+1)]
    logZ = 0

    if s.name == 'gibbs':
        logs_w = []

        for j in range(ais_num_samples):
            logw = 0
            for i in range(ais_num_dists):
                z = s.sample(c[i] * model.W, c[i] * model.b, 1)
                tmp = torch.einsum('bi,ij,bj->b',z[0,:model.z_dim].view(1,model.z_dim),
                                    model.W, z[0,model.z_dim:].view(1,model.z_dim)).item()
                tmp += torch.mm(z[0].view(1,2*model.z_dim), model.b.view(2*model.z_dim,1)).item()

                tmp = tmp * (c[i] - c[i+1])
                logw += tmp
            logs_w.append(logw)

        logs_w = [math.exp(lw) for lw in logs_w]
        logw = math.log(sum(logs_w)/ais_num_samples)

        logZ = logw + math.log(2.0) * model.z_dim * 2

#    print("log Z =", logZ)

    return logZ

# Model evaluation
def evaluate_model(model, data_loader, add_logZ=True):
    model.eval()
    total_size = 0
    total_loss = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            x_size = x.size(0)
            total_size += x_size
            x = x.to(device).view(-1, image_size)

            x_reconst, ent, p_ph = model(x)

            # Reconstruction loss
            reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')

            # Calculation of logZ
            if add_logZ:
                logZ = calc_logZ(model)
            else:
                logZ = 0.
            #print("log Z =", logZ)

            # KL divergence
            kl_div = -ent + p_ph + logZ * x_size
            #print("entropy =", ent.item()/x_size)
            #print("pos term =", p_ph.item()/x_size)
            #print("kl div = ", kl_div.item()/x_size)

            # Total loss
            loss = reconst_loss + kl_div

            total_loss += loss.item()

    # Average loss over dataset
    av_loss = total_loss / total_size

    return av_loss
