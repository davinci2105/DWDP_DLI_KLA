import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import gdown
from PIL import Image
import numpy as np

# Define your denoising model here
import segmentation_models_pytorch as smp

# AttentionUNet class definition
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

# Function to calculate PSNR
def calculate_psnr(prediction, target, mask, only_defect):
    if only_defect:
        mask = mask.to(torch.bool)
        mask_target = target[mask]
        mask_prediction = prediction[mask]
    else:
        mask_prediction = prediction
        mask_target = target

    mse = F.mse_loss(mask_prediction, mask_target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # Normalizing by 1.0 since images are in [0, 1] range
    return psnr.item()

# SSIM class to calculate SSIM between images
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, prediction, target, mask, only_defect):
        x = self.refl(target)
        y = self.refl(prediction)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        if only_defect:
            return (torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)) * mask
        else:
            return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

# Dataset class to handle denoising dataset
class DenoisingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self._collect_image_paths()

    def _collect_image_paths(self):
        for category in os.listdir(self.root_dir):
            category_path = os.path.join(self.root_dir, category)
            if not os.path.isdir(category_path):
                continue
            for dataset_type in ['Val']:  # Change to 'test' here for test set
                dataset_dir = os.path.join(category_path, dataset_type)
                if not os.path.isdir(dataset_dir):
                    continue
                gt_clean_dir = os.path.join(dataset_dir, 'GT_clean_image')
                defect_mask_dir = os.path.join(dataset_dir, 'Defect_mask')
                degraded_image_dir = os.path.join(dataset_dir, 'Degraded_image')

                for type_subfolder in os.listdir(gt_clean_dir):
                    type_clean_dir = os.path.join(gt_clean_dir, type_subfolder)
                    type_defect_mask_dir = os.path.join(defect_mask_dir, type_subfolder)
                    type_degraded_dir = os.path.join(degraded_image_dir, type_subfolder)

                    if not (os.path.isdir(type_clean_dir) and os.path.isdir(type_defect_mask_dir) and os.path.isdir(type_degraded_dir)):
                        continue

                    for file in os.listdir(type_clean_dir):
                        if file.endswith('.png') or file.endswith('.jpg'):
                            gt_clean_path = os.path.join(type_clean_dir, file)
                            base_name, _ = os.path.splitext(file)
                            defect_mask_file = f"{base_name}_mask.png"
                            defect_mask_path = os.path.join(type_defect_mask_dir, defect_mask_file)
                            degraded_image_path = os.path.join(type_degraded_dir, file)

                            if os.path.exists(defect_mask_path) and os.path.exists(degraded_image_path):
                                self.data.append({
                                    'file_name': file,
                                    'categories': category,
                                    'gt_clean': gt_clean_path,
                                    'defect_mask': defect_mask_path,
                                    'degraded_image': degraded_image_path
                                })

    def __len__(self):
        return len(self.data)

  

    def __getitem__(self, idx):
        paths = self.data[idx]
        filenames = paths['file_name']
        catg = paths['categories']
        
        # Read images using OpenCV
        gt_clean_img = cv2.imread(paths['gt_clean'])
        defect_mask_img = cv2.imread(paths['defect_mask'])
        degraded_img = cv2.imread(paths['degraded_image'])

        # Convert images to RGB format
        gt_clean_img = cv2.cvtColor(gt_clean_img, cv2.COLOR_BGR2RGB)
        defect_mask_img = cv2.cvtColor(defect_mask_img, cv2.COLOR_BGR2RGB)
        degraded_img = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB)
        
        # Convert images to PIL format
        gt_clean_img = Image.fromarray(gt_clean_img)
        defect_mask_img = Image.fromarray(defect_mask_img)
        degraded_img = Image.fromarray(degraded_img)
        
        # Apply transformations if specified
        if self.transform:
            gt_clean_img = self.transform(gt_clean_img)
            defect_mask_img = self.transform(defect_mask_img)
            degraded_img = self.transform(degraded_img)
        else:
            # Convert images to PyTorch tensors by default
            gt_clean_img = torch.tensor(gt_clean_img).permute(2, 0, 1).float() / 255.0
            defect_mask_img = torch.tensor(defect_mask_img).permute(2, 0, 1).float() / 255.0
            degraded_img = torch.tensor(degraded_img).permute(2, 0, 1).float() / 255.0
        
        return {
            'file_name': filenames,
            'categories': catg,
            'gt_clean': gt_clean_img,
            'defect_mask': defect_mask_img,
            'degraded_image': degraded_img
        }


# Load model function
def load_model(checkpoint_path, encoder_name='resnet34'):
    model = AttentionUNet(encoder_name=encoder_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset root and transform
dataset_root = "Dataset/Denoising_Dataset_train_val"  # Root directory of dataset
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize images to 1024x1024
    transforms.ToTensor(),            # Convert images to tensor
])

output_dir = 'Results_of_metrics'  # Directory for saving images
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
# Load model from checkpoint
file_id = '1gGiza9UsHM679TDlvn-1fhhu6V2Y0hS2'
url = f'https://drive.google.com/uc?id={file_id}'
checkpoint_path = f"{dataset_root}/Model/checkpoint_epoch_18.pth"

if not os.path.exists(checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    gdown.download(url, checkpoint_path, quiet=False)

model = load_model(checkpoint_path, encoder_name='resnet34')

# Instantiate dataset and create DataLoader
dataset = DenoisingDataset(dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

psnr_values = {}
ssim_values = {}
calculate_ssim = SSIM()

# Evaluation loop
count = 0
for batch in dataloader:
    filename = batch['file_name'][0]
    category = batch['categories'][0]
    gt_clean_batch = batch['gt_clean'].to('cuda' if torch.cuda.is_available() else 'cpu')
    defect_mask_batch = batch['defect_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
    degraded_image_batch = batch['degraded_image'].to('cuda' if torch.cuda.is_available() else 'cpu')

    # Model prediction
    with torch.no_grad():
        predicted = model(degraded_image_batch)

    # PSNR calculation
    psnr_value = calculate_psnr(predicted, gt_clean_batch, defect_mask_batch, only_defect=True)
    psnr_values[category] = psnr_values.get(category, []) + [psnr_value]

    # SSIM calculation
    ssim_value = calculate_ssim(predicted, gt_clean_batch, defect_mask_batch, only_defect=True)
    ssim_value = 1 - ssim_value.mean()
    ssim_values[category] = ssim_values.get(category, []) + [ssim_value.item()]

    # Save concatenated images
    count += 1
    if count % 5 == 0:
        save_path = os.path.join(output_dir, f"{category}_{count}_{filename}")
        concatenated_image = torch.cat([predicted[0], gt_clean_batch[0]], dim=2)
        save_image(concatenated_image, save_path)

# Calculate and print average PSNR and SSIM values
avg_psnr = {category: np.mean(psnr) for category, psnr in psnr_values.items()}
mean_ssim = {category: np.mean(ssim) for category, ssim in ssim_values.items()}
print("Average PSNR per category:", avg_psnr)
print("Average SSIM per category:", mean_ssim)

print("Total Average PSNR:", sum(avg_psnr.values()) / len(avg_psnr))
print("Total Average SSIM:", sum(mean_ssim.values()) / len(mean_ssim))
