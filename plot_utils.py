import matplotlib.pyplot as plt
import numpy as np
from inference import *

def plot_sample_trajectory(init_cond, sol, print_freq):
    
    sample_idx = np.random.randint(0, init_cond.shape[0])
    t_steps = sol.shape[-1]
    plt.figure(figsize=(20,4))

    plt.subplot(1,print_freq,1)
    plt.title('Initial Condition')
    plt.imshow(init_cond[sample_idx,:,:], cmap='jet')
    plt.colorbar()
    for i in range(1, print_freq):
        plt.subplot(1,print_freq,i+1)
        plt.title(f'Time step {i* (t_steps//print_freq)}')
        plt.imshow(sol[sample_idx,:,:, i* (t_steps//print_freq)], cmap='jet')
        plt.colorbar()
    plt.tight_layout()
    plt.show()

    return

def plot_batch_trainloader(train_loader, count):
    counter = 0
    field_t, field_t1, init_cond, time_t, _ = next(iter(train_loader))
    for counter in range(count):
        plt.figure(figsize=(8,4))
        plt.subplot(1,3,1)
        plt.imshow(init_cond[counter,0,:,:], cmap='viridis')
        plt.title(f'Initial Condition')
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(field_t[counter,0,:,:], cmap='viridis')
        plt.title(f'Field at {time_t[counter].item():.2f}')
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.imshow(field_t1[counter,0,:,:], cmap='viridis')
        plt.title(f'Field at {time_t[counter].item():.2f} + dt')
        plt.colorbar()
        plt.show()
        counter += 1
        if counter >= count:
            break
    return

def plot_loss_curve(train_loss):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def plot_reconstruction(model, sample_field_t, time_t, N_ITO_STEPS, num_plots, is_implicit, is_merge, device):
    
    
    sample_inputs = sample_field_t[:num_plots,:,:,:]
    recon_field, _  = inr_inference(model, sample_field_t, N_ITO_STEPS, is_implicit, is_merge, device)
    sample_field_t = sample_inputs.squeeze(1).cpu().numpy()

    _, axes = plt.subplots(num_plots, 2, figsize=(8, 4 * num_plots))
    for i in range(num_plots):
        axes[i, 0].imshow(sample_field_t[i], cmap='viridis')
        axes[i, 0].set_title(f'Original Field at {time_t[i].item():.2f}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon_field[i], cmap='viridis')
        axes[i, 1].set_title(f'Reconstructed Field at {time_t[i].item():.2f}')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

    return

def plot_rollout(true_fields, pred_fields, print_freq):
    fig, axes = plt.subplots(2, print_freq, figsize=(4 * print_freq, 6))
    
    for i in range(print_freq):
        axes[0, i].imshow(true_fields[i], cmap='viridis')
        axes[0, i].set_title(f'True (t={i* (len(true_fields)//print_freq)})')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(pred_fields[i], cmap='viridis')
        axes[1, i].set_title(f'Pred (t={i* (len(pred_fields)//print_freq)})')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

    return