import numpy as np
import matplotlib.pyplot as plt
from encoder import CNNEncoderBlock
from inr_wrapper import INRWrapper
from decoder import SynthesisNet, ModulationNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plot_utils import *
from data_utils import get_dataloader, generate_transformer_dataset
from inr_train import train_loop

PLOT = False ; TRAIN = False

# Load Data
init_cond = np.load('ns_2d_initialcond.npy')
sol = np.load('ns_2d_solutions.npy')
time = np.load('ns_2d_time.npy')

#print shapes
print('Initial Condition shape:', init_cond.shape)
print('Solution shape:', sol.shape)
print('Time shape:', time.shape)
nx = ny = init_cond.shape[1]

# plot a sample trajectory
if PLOT:
    plot_sample_trajectory(init_cond, sol, 5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
# Prepare DataLoader
train_loader, val_loader, test_loader = get_dataloader(
    sol_field=torch.tensor(sol, dtype=torch.float32),
    initial_cond=torch.tensor(init_cond, dtype=torch.float32),
    time=torch.tensor(time, dtype=torch.float32),
    batch_size=batch_size,
    shuffle=True
)

#visualize a batch
if PLOT:
    plot_batch_trainloader(train_loader, 5)

#hyperparameters
syn_features = [2,16,32,64,128,1]
mod_features = [int(2*i) for i in syn_features[:-1]]
encoder_channels = [1,16,32,32,64,128]
exp_latent_size = 256 ; imp_latent_size = None
num_epochs = 500 ; sampled_pts = 2048

# Initialize Model
dataloader_size = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
print(f"Dataloader size: {dataloader_size} samples")
cnn_encoder = CNNEncoderBlock(channels=encoder_channels, latent_size=exp_latent_size).to(device)
synthesis_net = SynthesisNet(features=syn_features).to(device)
modulation_net = ModulationNet(features=mod_features, z_emb_size= exp_latent_size).to(device)
model = INRWrapper(synthesis_net, modulation_net, cnn_encoder, dataloader_size, imp_latent_size, False, False).to(device)

# Define Loss function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(synthesis_net.parameters(), lr=5e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6)

# Training Loop
if TRAIN:
    model, train_loss = train_loop(model, train_loader, criterion, optimizer, scheduler, sampled_pts, num_epochs, device)
    plot_loss_curve(train_loss)
else:
    checkpoint = torch.load('saved_wts/inr_explicit_256.pth', map_location=device)
    model.load_state_dict(checkpoint)

#Reconstruction on sample test input 
sample_field_t, _, _, time_t, _ = next(iter(test_loader))
N_ITO_STEPS = 200 ; num_plots = 4
if PLOT:
    plot_reconstruction(model, sample_field_t, time_t, N_ITO_STEPS, num_plots, False, False, device)

#Compute relative error on a sample batch from test data
sample_inputs = sample_field_t[:batch_size,:,:,:]
recon_field, _  = inr_inference(model, sample_field_t, N_ITO_STEPS, False, False, device)
sample_field_t = sample_inputs.squeeze(1).cpu().numpy()
relative_error = np.mean((recon_field - sample_field_t)**2) / np.mean(sample_field_t**2)
print('relative error:',relative_error)







# =====================================
# I tried experimenting with training a transformer for learning the solution rollout in the latent space. The results aren't so great for now. So, commenting this part out

# Generate Transformer Dataset
# latent_seq_dataset = generate_transformer_dataset(model, init_cond, sol, train_loader.dataset, device)
# # Load the data
# torch.save(latent_seq_dataset, 'transformer_dataset.pt')
# print(f"Loaded dataset shape: {latent_seq_dataset.shape}")
    