import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Custom dataset for handling paired images (degraded, defect_mask, ground_truth)
class PairedImageDataset(Dataset):
    def __init__(self, degraded_dir, defect_mask_dir, ground_truth_dir, transform=None, mask_transform=None):
        self.degraded_dir = degraded_dir
        self.defect_mask_dir = defect_mask_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform          # Transformation for images (degraded, ground_truth)
        self.mask_transform = mask_transform  # Separate transformation for masks

        # List all images from the degraded directory
        self.image_names = sorted(os.listdir(degraded_dir))
        
        # Check for file matching in defect_mask and ground_truth folders
        self.image_names = [img for img in self.image_names if self._check_matching_files(img)]

    def _check_matching_files(self, img_name):
        """Ensure that the same image exists in all three folders."""
        defect_mask_img = os.path.join(self.defect_mask_dir, img_name)
        ground_truth_img = os.path.join(self.ground_truth_dir, img_name)
        
        if not os.path.exists(defect_mask_img):
            print(f"Missing defect_mask image for: {img_name}")
            return False
        if not os.path.exists(ground_truth_img):
            print(f"Missing ground_truth image for: {img_name}")
            return False
        return True

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Load the images
        degraded_image_path = os.path.join(self.degraded_dir, self.image_names[idx])
        defect_mask_path = os.path.join(self.defect_mask_dir, self.image_names[idx])
        ground_truth_path = os.path.join(self.ground_truth_dir, self.image_names[idx])

        degraded_image = Image.open(degraded_image_path)
        defect_mask_image = Image.open(defect_mask_path)
        ground_truth_image = Image.open(ground_truth_path)

        # Apply transformations (if any)
        if self.transform:
            degraded_image = self.transform(degraded_image)
            ground_truth_image = self.transform(ground_truth_image)

        if self.mask_transform:
            defect_mask_image = self.mask_transform(defect_mask_image)

        # Return a dictionary of the three images
        return {
            'degraded': degraded_image,
            'defect_mask': defect_mask_image,
            'ground_truth': ground_truth_image
        }

# Function to create DataLoaders for train and validation sets
def get_dataloaders(base_dir, batch_size=32):
    # Define the directories
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    # Define subfolder paths
    train_degraded = os.path.join(train_dir, 'degraded')
    train_defect_mask = os.path.join(train_dir, 'defect_mask')
    train_ground_truth = os.path.join(train_dir, 'ground_truth')

    val_degraded = os.path.join(val_dir, 'degraded')
    val_defect_mask = os.path.join(val_dir, 'defect_mask')
    val_ground_truth = os.path.join(val_dir, 'ground_truth')

    # Define transformations (only converting to tensor)
    image_transform = transforms.Compose([
        transforms.ToTensor()  # Convert images to tensors
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()  # Convert mask to tensor
    ])

    # Create datasets
    train_dataset = PairedImageDataset(train_degraded, train_defect_mask, train_ground_truth, transform=image_transform, mask_transform=mask_transform)
    val_dataset = PairedImageDataset(val_degraded, val_defect_mask, val_ground_truth, transform=image_transform, mask_transform=mask_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
