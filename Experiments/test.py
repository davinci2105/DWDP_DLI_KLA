from torchmetrics.functional import structural_similarity_index_measure as ssim
import sys
sys.path.append('/root/DWDP/')
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import calculate_psnr
from model import AttentionUNet
from dataset import loaders

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def visualize_test_results(model, test_loader, num_samples=5):
    model.eval()
    model.to(device)
    
    samples_visualized = 0
    with torch.no_grad():
        for degraded_img, mask, ground_truth in test_loader:
            degraded_img, mask, ground_truth = degraded_img.to(device), mask.to(device), ground_truth.to(device)
            
            # Get model output
            output = model(degraded_img)
            
            # Calculate PSNR and SSIM for each image in the batch
            for i in range(degraded_img.size(0)):
                psnr_value = calculate_psnr(output[i], ground_truth[i])
                
                # Add batch dimension for SSIM calculation
                degraded_img_batched = degraded_img[i].unsqueeze(0)  # Shape: [1, C, H, W]
                ground_truth_batched = ground_truth[i].unsqueeze(0)  # Shape: [1, C, H, W]
                output_batched = output[i].unsqueeze(0)              # Shape: [1, C, H, W]
                
                # Calculate SSIM between degraded and ground truth
                ssim_degraded_gt = ssim(degraded_img_batched, ground_truth_batched, data_range=1.0)
                
                # Calculate SSIM between degraded and output
                ssim_degraded_output = ssim(degraded_img_batched, output_batched, data_range=1.0)

                # Convert tensors to NumPy arrays for visualization
                degraded_np = degraded_img[i].cpu().numpy().transpose(1, 2, 0)
                mask_np = mask[i].cpu().numpy().squeeze()
                ground_truth_np = ground_truth[i].cpu().numpy().transpose(1, 2, 0)
                output_np = output[i].cpu().numpy().transpose(1, 2, 0)

                # Visualize the images
                plt.figure(figsize=(20, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(np.clip(degraded_np, 0, 1))
                plt.title('Degraded Image')
                plt.axis('off')
                
                plt.subplot(1, 4, 2)
                plt.imshow(mask_np, cmap='gray')
                plt.title('Mask')
                plt.axis('off')
                
                plt.subplot(1, 4, 3)
                plt.imshow(np.clip(ground_truth_np, 0, 1))
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(1, 4, 4)
                plt.imshow(np.clip(output_np, 0, 1))
                plt.title(f'Model Output\nPSNR: {psnr_value:.2f} dB\nSSIM (Degraded-GT): {ssim_degraded_gt:.2f}\nSSIM (Degraded-Output): {ssim_degraded_output:.2f}')
                plt.axis('off')

                plt.show()

                samples_visualized += 1
                if samples_visualized >= num_samples:
                    return  # Stop after visualizing the requested number of samples

# Load model from a specific checkpoint in a directory
def load_model_from_checkpoint(directory, checkpoint_name, encoder_name='resnet34'):
    # Create model
    model = AttentionUNet(encoder_name=encoder_name, pretrained=False).to(device)
    
    # Path to the checkpoint
    checkpoint_path = os.path.join(directory, checkpoint_name)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found in directory: {directory}")
    
    return model

# Define the directory and choose a specific checkpoint
model_directory = 'Model'  # Set your model directory path here
checkpoint_name = 'checkpoint_epoch_18.pth'  # Specify the exact checkpoint file name here
encoder_choice = 'resnet34'  # Options: 'resnet34', 'efficientnet-b0', etc.

# Load the model from the specified checkpoint
best_model_resnet = load_model_from_checkpoint(model_directory, checkpoint_name, encoder_name=encoder_choice)

# Load the test loader
_, _, test_loader = loaders()

# Visualize test results
visualize_test_results(model=best_model_resnet, test_loader=test_loader, num_samples=10)
