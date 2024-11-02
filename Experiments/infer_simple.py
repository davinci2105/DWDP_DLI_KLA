import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn as nn
import segmentation_models_pytorch as smp
import gdown

class AttentionUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=True):
        super(AttentionUNet, self).__init__()
        # Use a pre-trained encoder (either ResNet or EfficientNet)
        self.model = smp.Unet(
            encoder_name=encoder_name,        # Choose the encoder ('resnet34', 'efficientnet-b3', etc.)
            encoder_weights='imagenet' if pretrained else None,  # Use ImageNet weights if pretrained
            in_channels=3,                    # Number of input channels (RGB images)
            classes=3,                        # Number of output channels (RGB output)
            decoder_attention_type='scse'     # Use spatial and channel-wise attention
        )
        
    def forward(self, x):
        return self.model(x)


# Define the transformation 
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load a specific model checkpoint
def load_model(checkpoint_path, encoder_name='resnet34'):
    model = AttentionUNet(encoder_name=encoder_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate PSNR and SSIM
def calculate_metrics(output, ground_truth):
    psnr_value = psnr(output, ground_truth, data_range=1.0)
    ssim_value = ssim(output, ground_truth, data_range=1.0)
    return psnr_value.item(), ssim_value.item()

# Inference function with mask and ground truth SSIM
def infer_and_evaluate(model, input_folder, ground_truth_folder=None, mask_folder=None, save_output_folder="output_results"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    os.makedirs(save_output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue  # Skip non-image files

        # Load and preprocess the degraded image
        degraded_img_path = os.path.join(input_folder, filename)
        degraded_img = Image.open(degraded_img_path).convert("RGB")
        degraded_img_tensor = transform(degraded_img).unsqueeze(0).to(device)

        # Model inference
        output = model(degraded_img_tensor).squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        
        # Save the output image
        output_img = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
        output_img.save(os.path.join(save_output_folder, f"output_{filename}"))

        # Initialize figure for visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Keeping it to 4 for simplicity
        
        # Display degraded image
        axes[0].imshow(degraded_img)
        axes[0].set_title("Degraded Image")
        axes[0].axis("off")

        # Display ground truth if available
        if ground_truth_folder:
            ground_truth_img_path = os.path.join(ground_truth_folder, filename)
            if os.path.exists(ground_truth_img_path):
                ground_truth_img = Image.open(ground_truth_img_path).convert("RGB")
                ground_truth_tensor = transform(ground_truth_img).unsqueeze(0).to(device)
                
                # Calculate PSNR and SSIM between output and ground truth
                psnr_value, ssim_value_output_gt = calculate_metrics(
                    torch.tensor(output).unsqueeze(0).permute(0, 3, 1, 2).to(device),
                    ground_truth_tensor
                )
                
                # Calculate SSIM between degraded and ground truth
                _, ssim_value_degraded_gt = calculate_metrics(
                    degraded_img_tensor, ground_truth_tensor
                )
                
                axes[1].imshow(ground_truth_img)
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
            else:
                psnr_value, ssim_value_output_gt, ssim_value_degraded_gt = None, None, None
                axes[1].axis("off")
                print(f"No ground truth available for {filename}. Skipping metric calculation.")
        else:
            psnr_value, ssim_value_output_gt, ssim_value_degraded_gt = None, None, None
            axes[1].axis("off")

        # Display mask if available
        if mask_folder:
            mask_img_path = os.path.join(mask_folder, filename)
            if os.path.exists(mask_img_path):
                mask_img = Image.open(mask_img_path).convert("L")
                axes[2].imshow(mask_img, cmap='gray')
                axes[2].set_title("Mask")
                axes[2].axis("off")
            else:
                axes[2].axis("off")
                print(f"No mask available for {filename}.")
        else:
            axes[2].axis("off")

        # Display model output with PSNR and SSIM (both degraded vs GT and output vs GT if available)
        axes[3].imshow(output_img)
        if psnr_value is not None and ssim_value_output_gt is not None and ssim_value_degraded_gt is not None:
            axes[3].set_title(
                f"Output\nPSNR: {psnr_value:.2f} dB\nSSIM (Output vs GT): {ssim_value_output_gt:.4f}\n"
                f"SSIM (Degraded vs GT): {ssim_value_degraded_gt:.4f}"
            )
        else:
            axes[3].set_title("Output")
        axes[3].axis("off")

        plt.show()

# Usage example:
# Define paths
# Define the Google Drive file ID and download path
file_id = '1gGiza9UsHM679TDlvn-1fhhu6V2Y0hS2'
url = f'https://drive.google.com/uc?id={file_id}'
checkpoint_path = 'Model/checkpoint_epoch_18.pth'

# Check if the checkpoint file exists, if not, download it
if not os.path.exists(checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    print("Checkpoint file not found locally. Downloading from Google Drive...")
    gdown.download(url, checkpoint_path, quiet=False)
    print("Download complete.")

# Now load the checkpoint with torch.load
checkpoint = torch.load(checkpoint_path)

# Example: Load the model's state_dict
# Assuming `model` is your model instance
# model.load_state_dict(checkpoint['model_state_dict'])
print("Checkpoint loaded successfully.")
checkpoint_path = 'Model/checkpoint_epoch_18.pth'  # Specify the exact path to the desired checkpoint


## USER INPUT HERE , 
input_folder = 'structured_data/val/degraded'
ground_truth_folder = 'structured_data/val/ground_truth'  # Set to None if not available
mask_folder = 'structured_data/val/defect_mask'  # Include mask if available


save_output_folder = 'output_results'

# Load the model
model = load_model(checkpoint_path, encoder_name='resnet34')

# Perform inference and evaluation
infer_and_evaluate(model, input_folder, ground_truth_folder, mask_folder, save_output_folder)
