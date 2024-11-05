import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.utils import save_image
import gdown
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

# Define the AttentionUNet class
class AttentionUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=True):
        super(AttentionUNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=3,
            decoder_attention_type='scse'
        )

    def forward(self, x):
        return self.model(x)

# Load model function
def load_model(checkpoint_path, encoder_name='resnet34'):
    model = AttentionUNet(encoder_name=encoder_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define the dataset class
class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.noise_dir = os.path.join(root_dir, 'Noise')
        self.gt_dir = os.path.join(root_dir, 'Clean')
        self.transform = transform
        self.image_files = sorted(os.listdir(self.noise_dir))
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        noise_path = os.path.join(self.noise_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        noise_image = cv2.imread(noise_path)
        gt_image = cv2.imread(gt_path)
        
        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            noise_image = self.transform(noise_image)
            gt_image = self.transform(gt_image)
        else:
            noise_image = torch.tensor(noise_image).permute(2, 0, 1).float() / 255.0
            gt_image = torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0

        return {'noise': noise_image, 'gt': gt_image, 'filename': filename}

# Set dataset root, transformations, and output directory
dataset_root = "Dataset/new_data"
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
output_dir = 'Visualization_Results'
os.makedirs(output_dir, exist_ok=True)

# Load the model checkpoint from Google Drive if not already downloaded
file_id = '1gGiza9UsHM679TDlvn-1fhhu6V2Y0hS2'
checkpoint_path = f"{dataset_root}/Model/checkpoint_epoch_18.pth"
if not os.path.exists(checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    gdown.download(f'https://drive.google.com/uc?id={file_id}', checkpoint_path, quiet=False)

model = load_model(checkpoint_path)

# Initialize dataset and DataLoader
dataset = PairedImageDataset(root_dir=dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Evaluation loop
count = 0
for batch in dataloader:
    filename = batch['filename'][0]
    noise_image = batch['noise'].to('cuda' if torch.cuda.is_available() else 'cpu')
    gt_image = batch['gt'].to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        predicted = model(noise_image)

    # Save concatenated images every 5 samples
    count += 1
    if count % 5 == 0:
        save_path = os.path.join(output_dir, filename)
        concatenated_image = torch.cat([predicted[0], gt_image[0]], dim=2)
        save_image(concatenated_image, save_path)

print("Image processing completed and results saved.")
