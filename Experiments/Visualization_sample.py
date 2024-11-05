
# from 'model_folder' import model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Root directory containing the 'Noise' and 'Clean' subdirectories.
            transform (callable, optional): Optional transform to apply to both images.
        """
        self.root_dir = root_dir
        self.noise_dir = os.path.join(root_dir, 'Noise')
        self.gt_dir = os.path.join(root_dir, 'Clean')
        self.transform = transform

        # List of all image files in the Noise directory
        self.image_files = sorted(os.listdir(self.noise_dir))
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the filename and construct the paths for both Noise and GT images
        filename = self.image_files[idx]
        noise_path = os.path.join(self.noise_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        # Load images
        noise_image = cv2.imread(noise_path)
        gt_image = cv2.imread(gt_path)
        
        # Convert BGR (OpenCV format) to RGB
        noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            noise_image = self.transform(noise_image)
            gt_image = self.transform(gt_image)
        else:
            # Default transformation: Convert to PyTorch tensors and normalize to [0,1]
            noise_image = torch.tensor(noise_image).permute(2, 0, 1).float() / 255.0
            gt_image = torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0

        return {'noise': noise_image, 'gt': gt_image, 'filename': filename}


dataset_root = "Denoising_Dataset_Test_Visual"  # Update this with the actual path to your data directory

output_dir='Results'  ##### OUTPUT DIRECTORY  ###############

transform = transforms.ToTensor()  # Define any additional transformations if needed


dataset = PairedImageDataset(root_dir=dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

count=0
for batch in dataloader:
    noise_images = batch['noise']
    gt_images = batch['gt']
    filenames = batch['filename']
    count=count+1
    print(count)
    # predicted= ## WRITE CODE TO PASS INPUT TO THE MODEL
    
    
    save_path=os.path.join(output_dir,filenames[0])
    concatenated_image=torch.cat([noise_images[0],gt_images[0]],dim=2)
    save_image( concatenated_image,save_path)
    
    
    
   

