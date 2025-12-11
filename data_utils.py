import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from inference import *

class NSDataset(Dataset):
    def __init__(self, sol_field, initial_cond, time):
        self.sol_field = sol_field # [N, H, W, T]
        self.initial_cond = initial_cond
        self.time = time
        self.tsteps = sol_field.shape[-1] + 1 # including initial condition
        self.nconds = sol_field.shape[0]

    def __len__(self):
        return int(self.nconds * self.tsteps)

    def __getitem__(self, idx):
        n = idx // self.tsteps
        t = idx % self.tsteps
        if t == 0:
            field_t = self.initial_cond[n, :, :]
            field_t1 = self.sol_field[n, :, :, t]
        else:
            field_t = self.sol_field[n, :, :, min(t-1, self.tsteps-3)]
            field_t1 = self.sol_field[n, :, :, min(t, self.tsteps-2)]
        
        init_cond = self.initial_cond[n, :, :]
        time_t = self.time[min(t,self.tsteps-2)] / self.time[-1]  # Normalize time
        
        return field_t.unsqueeze(0), field_t1.unsqueeze(0), init_cond.unsqueeze(0), time_t, idx
    
def get_dataloader(sol_field, initial_cond, time, batch_size=32, shuffle=True):
    dataset = NSDataset(sol_field, initial_cond, time)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def generate_transformer_dataset(model, init_cond, sol, train_set, device, n_ito_steps=200):
    
    model.eval()

    for p in model.parameters():
        p.requires_grad = False
        
    train_indices_set = set(train_set.indices)
    print(f"Recovered {len(train_indices_set)} training indices for Lookup.")

    N_sims = init_cond.shape[0]
    T_total = sol.shape[-1] + 1
    
    all_sequences = []
    
    print(f"Generating sequences for {N_sims} simulations...")
    for n in range(N_sims):
        sim_latents = []
        
        for t in tqdm(range(T_total), desc=f"Sim {n+1}/{N_sims}", leave=False):
            
            global_idx = n * T_total + t
            
            if t == 0:
                field_np = init_cond[n]
            else:
                field_np = sol[n, :, :, t-1]
                
            field_t = torch.tensor(field_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                z_explicit = model.encoder(field_t) # [1, latent_size]

            if global_idx in train_indices_set:
                with torch.no_grad():
                    idx_tensor = torch.tensor([global_idx], device=device)
                    z_imp = model.implicit_embedding(idx_tensor)
            else:
                _, z_imp = inr_inference(model, field_t, n_ito_steps, device)

            z_combined = torch.cat([z_explicit, z_imp], dim=-1)
            sim_latents.append(z_combined.cpu())

        all_sequences.append(torch.stack(sim_latents).squeeze(1))

    transformer_dataset = torch.stack(all_sequences)
    print(f"Generation Complete. Final Tensor Shape: {transformer_dataset.shape}")
    
    return transformer_dataset

    
def latent_dataloader(dataset, batch_size=16, shuffle=True):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class LatentSeqDataset(Dataset):
    def __init__(self, data_path, device, window_size=5):

        self.latent_seq = torch.load(data_path, map_location=device) 
        
        self.N_sims, self.T_total, self.D = self.latent_seq.shape
        self.window_size = window_size
        
        self.samples_per_sim = self.T_total - self.window_size
        

    def __len__(self):
        return self.N_sims * self.samples_per_sim

    def __getitem__(self, idx):

        sim_idx = idx // self.samples_per_sim
        start_t = idx % self.samples_per_sim
        
        end_t = start_t + self.window_size + 1
        
        chunk = self.latent_seq[sim_idx, start_t:end_t, :] 
        
        src = chunk[:-1, :] 
        tgt = chunk[1:, :]   
        
        return src, tgt