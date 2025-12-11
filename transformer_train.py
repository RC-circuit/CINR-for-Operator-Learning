import torch.optim as optim
import torch

def train_loop(model, train_loader, optimizer, criterion, scheduler, epochs, device):
    print("Starting Transformer Training...")
    train_loss = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            prediction = model(src) 

            loss = criterion(prediction, tgt)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Current LR: {current_lr}")
        
        train_loss.append(total_loss / len(train_loader))
        scheduler.step(total_loss / len(train_loader))
    
    # Save the trained transformer
    torch.save(model.state_dict(), 'latent_transformer_model.pth')
    print("Training Complete.")
    
    return model, train_loss