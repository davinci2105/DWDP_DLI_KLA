import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# Load dataset paths
degraded_folder = '/root/DLI_KLA/structured_data/train/degraded'
mask_folder = '/root/DLI_KLA/structured_data/train/defect_mask'
ground_truth_folder = '/root/DLI_KLA/structured_data/train/ground_truth'

# Custom Dataset Class
class DenoisingDataset(Dataset):
    def __init__(self, degraded_folder, mask_folder, ground_truth_folder, transform=None):
        self.degraded_images = sorted([os.path.join(degraded_folder, img) for img in os.listdir(degraded_folder)])
        self.masks = sorted([os.path.join(mask_folder, img) for img in os.listdir(mask_folder)])
        self.ground_truths = sorted([os.path.join(ground_truth_folder, img) for img in os.listdir(ground_truth_folder)])
        self.transform = transform

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        degraded_img = Image.open(self.degraded_images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        ground_truth = Image.open(self.ground_truths[idx]).convert('RGB')
        
        if self.transform:
            degraded_img = self.transform(degraded_img)
            mask = self.transform(mask)
            ground_truth = self.transform(ground_truth)

        return degraded_img, mask, ground_truth

# Custom function for adaptive padding
def pad_to_match_max_size(images):
    # Find the maximum width and height in the batch
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    # Make max_height and max_width divisible by 32
    max_height = ((max_height + 31) // 32) * 32
    max_width = ((max_width + 31) // 32) * 32

    # Pad each image to match the maximum dimensions that are divisible by 32
    padded_images = []
    for img in images:
        channels, height, width = img.shape
        pad_height = max_height - height
        pad_width = max_width - width
        padded_img = nn.functional.pad(img, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_images.append(padded_img)

    return torch.stack(padded_images)

# Custom collate function to handle images of different sizes
def custom_collate_fn(batch):
    degraded_imgs, masks, ground_truths = zip(*batch)

    # Pad images to the maximum size in the batch
    degraded_imgs = pad_to_match_max_size(degraded_imgs)
    masks = pad_to_match_max_size(masks)
    ground_truths = pad_to_match_max_size(ground_truths)

    return degraded_imgs, masks, ground_truths

# DataLoader function
def loaders():
    # Data transformations (Note: No resizing here)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Only convert to tensor without resizing
    ])

    # Create dataset
    dataset = DenoisingDataset(degraded_folder, mask_folder, ground_truth_folder, transform=transform)

    # Split dataset into train, validation, and test (70% train, 20% validation, 10% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader
