import torch

# Hyperparameters

data_dir = 'data'
mini_sets = False # whether to use mini datasets for calculating total losses at the end of the epoch / learning process
batch_size = 128
image_size = 784
h_dim = 400
z_dim = 10  # number of "visible" units of RBM (equals to the number of "hidden" units)
numh = 1 # number of layers in hierarchical posterior, 1 or 2
num_epochs = 100
learning_rate = 1e-3
beta = 7.0
eps = 1e-8
gc_num = 30 # number of sampler Gibbs cycle
num_samples_pb = batch_size # number of samples per batch during training
ais_num_samples = 10 # number of samples used in annealed importance sampling for log Z estimation
ais_num_dists = 10 # number of distributions used in ais for log Z estimation
results_dir='results/'

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Sampler
from sampler import GSampler as Sampler
