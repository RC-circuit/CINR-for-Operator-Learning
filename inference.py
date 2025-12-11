import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def inr_inference(model, sample_field_t, N_ITO_STEPS, is_implicit, is_merge, device):

    model.eval()
    sample_field_t = sample_field_t.to(device)

    B, _, H, W = sample_field_t.shape

    ys, xs = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )

    query_pts = torch.stack((xs, ys), dim=-1).view(-1, 2)
    for params in model.parameters():
        params.requires_grad = False
    if is_implicit or is_merge:
        z_new_dim = model.implicit_embedding.embedding_dim
        z_new = torch.randn(B, z_new_dim, device=device) * 1e-2
        z_new.requires_grad = True
        optimizer_ito = optim.Adam([z_new], lr=1e-3)
        criterion = nn.MSELoss()

        for step in range(N_ITO_STEPS):
            optimizer_ito.zero_grad()
            outputs = model.forward_ito(sample_field_t, query_pts, z_new)
            targets = sample_field_t[:, 0, :, :].reshape(-1, 1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ito.step()
            
            if (step + 1) % 25 == 0:
                print(f"ITO Step {step+1}/{N_ITO_STEPS}, Loss: {loss.item():.6f}")

        z_new = z_new.detach()
        outputs = model.forward_ito(sample_field_t, query_pts, z_new)
    else:
        outputs = model(sample_field_t, query_pts, None)
        z_new = None
    recon_field = outputs.view(B, H, W).cpu().numpy()

    return recon_field, z_new


def transformer_inference(transformer, z0, rollout_steps, window_size, device):
    
    transformer.eval()
    
    current_history = z0.to(device) # [1, 1, D]
    
    predictions = []
    
    with torch.no_grad():
        for i in range(rollout_steps):

            if current_history.shape[1] > window_size:
                input_seq = current_history[:, -window_size:, :]
            else:
                input_seq = current_history
            
            output = transformer(input_seq)
            
            next_latent = output[:, -1:, :]

            current_history = torch.cat([current_history, next_latent], dim=1)
            predictions.append(next_latent.cpu())
            
    return torch.cat(predictions, dim=1)

def combined_rollout(inr_model, transformer, init_field, N_ITO_STEPS, rollout_steps, window_size, device):
    inr_model.eval()
    init_field = init_field.unsqueeze(0).to(device)
    with torch.no_grad():
        z_explicit = inr_model.encoder(init_field)
    
    _, z_implicit = inr_inference(inr_model, init_field, N_ITO_STEPS, device)
    z0 = torch.cat([z_explicit, z_implicit], dim=-1).unsqueeze(1).to(device)
    
    latent_predictions = transformer_inference(transformer, z0, rollout_steps, window_size, device)
    predictions = []
    predictions.append(init_field.squeeze(0).cpu().numpy())
    for t in range(rollout_steps):
        z_t = latent_predictions[:, t, :].to(device)

        B, _, H, W = init_field.shape
        ys, xs = torch.meshgrid(
                torch.linspace(0, 1, H, device=device),
                torch.linspace(0, 1, W, device=device),
                indexing='ij'
            )

        query_pts = torch.stack((xs, ys), dim=-1).view(-1, 2)

        with torch.no_grad():
            output = inr_model.forward_ito(init_field, query_pts, z_t, full_embed=True)
            recon_field = output.view(B, H, W).cpu().numpy()
            predictions.append(recon_field)

    return np.concatenate(predictions, axis=0)