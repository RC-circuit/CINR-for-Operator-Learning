import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_sample_trajectory
import torch
import matplotlib.animation as animation
from data_utils import *
from encoder import *
from decoder import *
from inr_wrapper import *

# Load Data
init_cond = np.load('ns_2d_initialcond.npy')
sol = np.load('ns_2d_solutions.npy')
time = np.load('ns_2d_time.npy')

#print shapes
print('Initial Condition shape:', init_cond.shape)
print('Solution shape:', sol.shape)
print('Time shape:', time.shape)
nx = ny = init_cond.shape[1]

sample_idx = np.random.randint(0, init_cond.shape[0])
true_field = np.concatenate((np.expand_dims(init_cond[sample_idx,:,:], axis=0), np.transpose(sol[sample_idx, :, :, :],(2,0,1))), axis=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare DataLoader
train_loader, val_loader, test_loader = get_dataloader(
    sol_field=torch.tensor(sol, dtype=torch.float32),
    initial_cond=torch.tensor(init_cond, dtype=torch.float32),
    time=torch.tensor(time, dtype=torch.float32),
    batch_size=16,
    shuffle=True
)


#hyperparameters
syn_features = [2,16,32,64,128,1]
mod_features = [int(2*i) for i in syn_features[:-1]]
encoder_channels = [1,16,32,32,64,128]
exp_latent_size = 256 ; imp_latent_size = 256
num_epochs = 500 ; sampled_pts = 2048

# Initialize Model
dataloader_size = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
print(f"Dataloader size: {dataloader_size} samples")
cnn_encoder = CNNEncoderBlock(channels=encoder_channels, latent_size=exp_latent_size).to(device)
synthesis_net = SynthesisNet(features=syn_features).to(device)
modulation_net = ModulationNet(features=mod_features, z_emb_size= exp_latent_size).to(device)
model = INRWrapper(synthesis_net, modulation_net, cnn_encoder, dataloader_size, exp_latent_size, False, False).to(device)

checkpoint = torch.load('inr_explicit_256.pth', map_location=device)
model.load_state_dict(checkpoint)
true_field_tensor = torch.tensor(true_field, device=device)
print(true_field_tensor.shape)
pred_field, _ = inr_inference(model, true_field_tensor.unsqueeze(1), 200, False, False, device)
print(pred_field.shape)
def create_smooth_jet_gif(data, filename="prediction_evolution.gif", fps=15):
    
    vmin = data.min()
    vmax = data.max()
    
    # 3. Setup Figure
    # Removing whitespace/borders for a cleaner look
    fig = plt.figure(figsize=(6, 6), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # 4. Initialize Plot
    # interpolation='bicubic' creates the "fine", smooth look
    im = ax.imshow(data[0], cmap='jet', interpolation='bicubic', vmin=vmin, vmax=vmax)
    
    def update(frame_idx):
        im.set_data(data[frame_idx])
        return im,
    
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(data), 
        interval=1000/fps, 
        blit=True
    )
    
    ani.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print("Done.")


create_smooth_jet_gif(true_field, "GT.gif")
create_smooth_jet_gif(pred_field, "pred.gif")