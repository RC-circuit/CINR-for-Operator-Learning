import torch
import matplotlib.pyplot as plt
from inference import *

def train_loop(model, trainloader, criterion, optimizer, scheduler, sampled_pts, num_epochs, device, diagnosis = False, MODEL_SAVE_PATH = 'inr_mixed_256.pth'):
    print("Starting Training...")
    network_modules = {
    "CNN Encoder": model.encoder,
    "Modulation Network": model.modnet,
    "Synthesis Network": model.synnet
    }
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for params in model.parameters():
            params.requires_grad = True
        model.train() # Set the model to training mode
        for field_t, _, _, _, idx in trainloader:

            inputs = field_t.to(device)
            _, _, H, W = inputs.shape
            indices = idx.to(device)
            optimizer.zero_grad()

            query_pts = torch.rand(sampled_pts, 2, device=device)

            outputs = model(inputs, query_pts, indices)

            sampled_indices_x = (query_pts[:,0] * (W - 1)).long()
            sampled_indices_y = (query_pts[:,1] * (H - 1)).long()

            targets = inputs[:, 0, sampled_indices_y, sampled_indices_x].reshape(-1, 1).to(device)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if diagnosis:
            for submodel_name, module in network_modules.items():
                print(f"{submodel_name} gradient status:")
                for name, param in module.named_parameters():
                    if param.grad is not None:
                        print(f"  {name}: Max gradient value = {param.grad.abs().max():.6f}")
                    else:
                        print(f"  {name}: GRADIENT IS NONE (ERROR)")
 
        scheduler.step(running_loss / len(trainloader))
        avg_train_loss = running_loss / len(trainloader)

        if epoch % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Current LR: {current_lr}")
            
            sample_input = inputs[0:1, :, :, :]
            recon_field, _ = inr_inference(model, sample_input, 200, True, False, device)
            target_field = sample_input[0, 0, :, :].cpu().numpy()
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.imshow(target_field, cmap='viridis')
            plt.title('Target Field')
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(np.squeeze(recon_field), cmap='viridis')
            plt.title('Reconstructed Field')
            plt.colorbar()
            plt.suptitle(f'Epoch {epoch+1} Reconstruction')
            plt.show()

        train_loss.append(avg_train_loss)

    print('Finished Training')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Model state saved successfully to {MODEL_SAVE_PATH}")
    return model, train_loss