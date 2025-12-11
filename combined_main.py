import torch
import torch.nn as nn
from inr_wrapper import INRWrapper
from decoder import SynthesisNet, ModulationNet
from encoder import CNNEncoderBlock
from transformer import LatentTransformer
from plot_utils import *
from inference import *
import numpy as np

sol = np.load('ns_2d_solutions.npy')
init_cond = np.load('ns_2d_initialcond.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_idx = np.random.randint(0, init_cond.shape[0])
init_field = torch.tensor(init_cond[sample_idx,:,:], dtype=torch.float32).unsqueeze(0)

#hyperparameters
syn_features = [2,16,32,64,128,1]
mod_features = [int(2*i) for i in syn_features[:-1]]
encoder_channels = [1,16,32,32,64,64]
exp_latent_size = 128 ; imp_latent_size = 64
latent_dim = exp_latent_size + imp_latent_size
d_model = 256

# Initialize Model
dataloader_size = init_cond.shape[0] * (sol.shape[-1] + 1)
cnn_encoder = CNNEncoderBlock(channels=encoder_channels, input_size = init_cond.shape[1], latent_size=exp_latent_size).to(device)
synthesis_net = SynthesisNet(features=syn_features).to(device)
modulation_net = ModulationNet(features=mod_features, z_emb_size= int(exp_latent_size + imp_latent_size)).to(device)
inr_model = INRWrapper(synthesis_net, modulation_net, cnn_encoder, dataloader_size, imp_latent_size).to(device)
transformer_model = LatentTransformer(latent_dim, d_model=d_model).to(device)

inr_checkpoint = torch.load('inr_model.pth', map_location=device)
print("Model loaded from inr_model.pth")
inr_model.load_state_dict(inr_checkpoint)

inr_checkpoint = torch.load('latent_transformer_model.pth', map_location=device)
print("Model loaded from transformer_model.pth")
transformer_model.load_state_dict(inr_checkpoint)


pred_field = combined_rollout(inr_model, transformer_model, init_field, N_ITO_STEPS=200, rollout_steps=20, window_size=5, device=device)
true_field = np.concatenate((np.expand_dims(init_cond[sample_idx,:,:], axis=0), np.transpose(sol[sample_idx, :, :, :],(2,0,1))), axis=0)

plot_rollout(true_field, pred_field, print_freq=5)

