from data_utils import *
import torch
from transformer import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
from transformer_train import train_loop
from plot_utils import plot_loss_curve

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = LatentSeqDataset('transformer_dataset.pt', device, window_size=5)
train_loader, val_loader, test_loader = latent_dataloader(dataset, batch_size=16, shuffle=True)
print(f"Train Loader Size: {len(train_loader.dataset)} samples")
print(f"Validation Loader Size: {len(val_loader.dataset)} samples")
print(f"Test Loader Size: {len(test_loader.dataset)} samples")

# Hyperparameters
latent_dim = 192
d_model = 256
learning_rate = 5e-3
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup
model = LatentTransformer(latent_dim, d_model=d_model).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-4)

TRAIN = True
# Training Loop
if TRAIN:
    model, train_loss = train_loop(model, train_loader, optimizer, criterion, scheduler, epochs, device)
else:
    model.load_state_dict(torch.load('latent_transformer_model.pth', map_location=device))
    print("Model loaded from latent_transformer_model.pth")
    train_loss = []

if len(train_loss) > 0:
    plot_loss_curve(train_loss)
    