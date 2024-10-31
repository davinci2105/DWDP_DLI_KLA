import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from model import AttentionUNet
from dataset import DenoisingDataset, custom_collate_fn, loaders
from utils import WeightedLoss, calculate_psnr

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to train and evaluate the model
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=20, model_name='AttentionUNet', save_path='best_model.pth', resume_training=True, load_epoch=None):
    train_losses = []
    val_losses = []
    val_psnr_scores = []
    best_val_loss = float('inf')
    start_epoch = 0

    # Create a new folder for checkpoints with model name and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f'{model_name}_{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load checkpoint from a specific epoch if requested
    if load_epoch is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{load_epoch}.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from epoch {load_epoch} ({checkpoint_path})...")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            val_psnr_scores = checkpoint['val_psnr_scores']
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"Checkpoint for epoch {load_epoch} not found. Starting from scratch.")
    
    # Resume from the latest checkpoint if resume_training is True and no specific epoch is provided
    elif resume_training and os.path.exists(save_path):
        print(f"Loading checkpoint from {save_path}...")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_psnr_scores = checkpoint['val_psnr_scores']
        print(f"Resumed training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        model.train()
        epoch_loss = 0.0

        # Using tqdm for progress bar during training
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}", unit="batch") as pbar:
            for degraded_img, mask, ground_truth in train_loader:
                degraded_img, mask, ground_truth = degraded_img.to(device), mask.to(device), ground_truth.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = model(degraded_img)
                
                # Calculate weighted loss
                loss = criterion(output, ground_truth, mask)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

        # Average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        psnr_total = 0.0
        with torch.no_grad():
            for degraded_img, mask, ground_truth in val_loader:
                degraded_img, mask, ground_truth = degraded_img.to(device), mask.to(device), ground_truth.to(device)
                
                output = model(degraded_img)
                
                # Calculate validation loss
                val_loss += criterion(output, ground_truth, mask).item()
                
                # Calculate PSNR
                psnr_total += calculate_psnr(output, ground_truth)

        avg_val_loss = val_loss / len(val_loader)
        avg_psnr = psnr_total / len(val_loader)
        val_losses.append(avg_val_loss)
        val_psnr_scores.append(avg_psnr)

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Save checkpoint for this epoch in the new directory
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_psnr_scores': val_psnr_scores
        }, checkpoint_path)

        # Save the best model if validation improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        # Print epoch summary with flush=True
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f} dB", flush=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Plot validation PSNR
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_psnr_scores) + 1), val_psnr_scores, label='Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('Validation PSNR Over Epochs')
    plt.show()

# Choose encoder (ResNet or EfficientNet)
encoder_choice = 'resnet34'  # Options: 'resnet34', 'efficientnet-b0', 'efficientnet-b3', etc.
model = AttentionUNet(encoder_name=encoder_choice, pretrained=True).to(device)

# Freeze encoder layers (accessing the encoder through model.model.encoder)
for param in model.model.encoder.parameters():
    param.requires_grad = False

# Only train decoder and attention layers
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-5)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Custom Weighted Loss
criterion = WeightedLoss(base_loss=nn.MSELoss(), weight_factor=0.8)

train_loader,val_loader,_=loaders()
# Train the model
train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=20, model_name='AttentionUNet', save_path='best_model_mse_resnet.pth', load_epoch=12)
