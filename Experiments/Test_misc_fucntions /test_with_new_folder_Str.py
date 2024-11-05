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
from fpdf import FPDF
from collections import defaultdict
import time

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

# Define transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load model
def load_model(checkpoint_path, encoder_name='resnet34'):
    model = AttentionUNet(encoder_name=encoder_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Metric calculation with masking
def calculate_metrics_with_mask(output, ground_truth, mask):
    psnr_value = psnr(output, ground_truth, data_range=1.0)
    ssim_value = ssim(output, ground_truth, data_range=1.0)
    masked_output = output * mask
    masked_ground_truth = ground_truth * mask
    psnr_masked = psnr(masked_output, masked_ground_truth, data_range=1.0)
    ssim_masked = ssim(masked_output, masked_ground_truth, data_range=1.0)
    return psnr_value.item(), ssim_value.item(), psnr_masked.item(), ssim_masked.item()

# Plot and save average metrics
def plot_average_metrics(class_metrics, output_dir):
    classes = list(class_metrics.keys())
    psnr_whole_gt = [np.mean(class_metrics[cls]['psnr_whole_gt']) for cls in classes]
    ssim_whole_gt = [np.mean(class_metrics[cls]['ssim_whole_gt']) for cls in classes]
    psnr_whole_output = [np.mean(class_metrics[cls]['psnr_whole_output']) for cls in classes]
    ssim_whole_output = [np.mean(class_metrics[cls]['ssim_whole_output']) for cls in classes]
    psnr_masked_gt = [np.mean(class_metrics[cls]['psnr_masked_gt']) for cls in classes]
    ssim_masked_gt = [np.mean(class_metrics[cls]['ssim_masked_gt']) for cls in classes]
    psnr_masked_output = [np.mean(class_metrics[cls]['psnr_masked_output']) for cls in classes]
    ssim_masked_output = [np.mean(class_metrics[cls]['ssim_masked_output']) for cls in classes]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].bar(classes, psnr_whole_gt, color='lightblue', label="Degraded vs GT")
    axs[0, 0].bar(classes, psnr_whole_output, color='lightgreen', label="Output vs GT", alpha=0.7)
    axs[0, 0].set_title("Average PSNR per Class (Whole Image)")
    axs[0, 0].set_ylabel("PSNR (dB)")
    axs[0, 0].legend()

    axs[0, 1].bar(classes, ssim_whole_gt, color='salmon', label="Degraded vs GT")
    axs[0, 1].bar(classes, ssim_whole_output, color='orange', label="Output vs GT", alpha=0.7)
    axs[0, 1].set_title("Average SSIM per Class (Whole Image)")
    axs[0, 1].set_ylabel("SSIM")
    axs[0, 1].legend()

    axs[1, 0].bar(classes, psnr_masked_gt, color='lightblue', label="Masked Degraded vs GT")
    axs[1, 0].bar(classes, psnr_masked_output, color='lightgreen', label="Masked Output vs GT", alpha=0.7)
    axs[1, 0].set_title("Average PSNR per Class (Masked Region)")
    axs[1, 0].set_ylabel("PSNR (dB)")
    axs[1, 0].legend()

    axs[1, 1].bar(classes, ssim_masked_gt, color='salmon', label="Masked Degraded vs GT")
    axs[1, 1].bar(classes, ssim_masked_output, color='orange', label="Masked Output vs GT", alpha=0.7)
    axs[1, 1].set_title("Average SSIM per Class (Masked Region)")
    axs[1, 1].set_ylabel("SSIM")
    axs[1, 1].legend()

    plt.tight_layout()
    avg_metrics_path = os.path.join(output_dir, "average_metrics.png")
    plt.savefig(avg_metrics_path)
    plt.close()
    return avg_metrics_path

# Inference and evaluation function
def infer_and_evaluate(model, input_folder, ground_truth_folder, mask_folder, save_output_folder="output_results", n_images=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    class_metrics = defaultdict(lambda: {
        'psnr_whole_gt': [], 'ssim_whole_gt': [], 'psnr_whole_output': [], 'ssim_whole_output': [],
        'psnr_masked_gt': [], 'ssim_masked_gt': [], 'psnr_masked_output': [], 'ssim_masked_output': []
    })
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(save_output_folder, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare PDF with cover page
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Test Results for Denoising with Defect Preservation", ln=True, align='C')
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, "Project by: Sumeet, Divya, Gunjan, and Athira for Course EE5179 Deep Learning for Imaging", ln=True, align='C')
    pdf.ln(20)  # Space before starting image results

    processed_images = 0
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        if n_images is not None and processed_images >= n_images:
            break  # Stop processing if we've reached the limit

        class_name = filename.split('_')[0]

        degraded_img_path = os.path.join(input_folder, filename)
        ground_truth_img_path = os.path.join(ground_truth_folder, filename)
        mask_img_path = os.path.join(mask_folder, filename)

        degraded_img = Image.open(degraded_img_path).convert("RGB")
        degraded_img_tensor = transform(degraded_img).unsqueeze(0).to(device)

        output = model(degraded_img_tensor).squeeze().cpu().detach().numpy().transpose(1, 2, 0)
        output_img = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
        output_img_path = os.path.join(output_dir, f"output_{filename}")
        output_img.save(output_img_path)

        if os.path.exists(ground_truth_img_path):
            ground_truth_img = Image.open(ground_truth_img_path).convert("RGB")
            ground_truth_tensor = transform(ground_truth_img).unsqueeze(0).to(device)

            if os.path.exists(mask_img_path):
                mask_img = Image.open(mask_img_path).convert("L")
                mask_tensor = transform(mask_img).unsqueeze(0).to(device)
            else:
                mask_tensor = torch.ones_like(degraded_img_tensor[:, :1])

            psnr_whole, ssim_whole, psnr_masked, ssim_masked = calculate_metrics_with_mask(
                torch.tensor(output).unsqueeze(0).permute(0, 3, 1, 2).to(device),
                ground_truth_tensor, mask_tensor
            )
            psnr_degraded_gt, ssim_degraded_gt, psnr_degraded_masked, ssim_degraded_masked = calculate_metrics_with_mask(
                degraded_img_tensor, ground_truth_tensor, mask_tensor
            )

            # Store metrics
            class_metrics[class_name]['psnr_whole_gt'].append(psnr_degraded_gt)
            class_metrics[class_name]['ssim_whole_gt'].append(ssim_degraded_gt)
            class_metrics[class_name]['psnr_whole_output'].append(psnr_whole)
            class_metrics[class_name]['ssim_whole_output'].append(ssim_whole)
            class_metrics[class_name]['psnr_masked_gt'].append(psnr_degraded_masked)
            class_metrics[class_name]['ssim_masked_gt'].append(ssim_degraded_masked)
            class_metrics[class_name]['psnr_masked_output'].append(psnr_masked)
            class_metrics[class_name]['ssim_masked_output'].append(ssim_masked)

            # Display in PDF
            pdf.add_page()
            pdf.cell(200, 10, f"Results for {filename}", ln=True)
            pdf.cell(200, 10, f"PSNR (Whole): {psnr_whole:.2f} dB, SSIM (Whole): {ssim_whole:.4f}", ln=True)
            pdf.cell(200, 10, f"PSNR (Masked): {psnr_masked:.2f} dB, SSIM (Masked): {ssim_masked:.4f}", ln=True)
            pdf.image(degraded_img_path, x=10, y=50, w=40, h=40)
            pdf.image(ground_truth_img_path, x=60, y=50, w=40, h=40)
            pdf.image(mask_img_path, x=110, y=50, w=40, h=40)
            pdf.image(output_img_path, x=160, y=50, w=40, h=40)

        processed_images += 1

    # Plot average metrics at the end
    avg_metrics_path = plot_average_metrics(class_metrics, output_dir)
    pdf.add_page()
    pdf.image(avg_metrics_path, x=10, y=10, w=190)

    pdf_file = os.path.join(output_dir, "report.pdf")
    pdf.output(pdf_file)
    print(f"Report saved at: {pdf_file}")

import shutil

# Function to rearrange and rename files as per desired structure
def rearrange_and_rename_files(src_dir, dest_dir):
    """
    Rearranges the folder structure from:
    - src_dir/Class/Train/[Defect_mask/defect_type/images, Degraded_image/defect_type/images, clean_image/defect_type/images]
    
    To:
    - dest_dir/Train_or_Val/class_name_defect_type/Defect/images, Degraded/images, Ground_Truth/images
    
    Args:
        src_dir (str): The source directory of the original structure.
        dest_dir (str): The destination directory for the new structure.
    """
    # Loop through each class in the source directory
    for class_name in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_name)
        if os.path.isdir(class_path):
            # Process both Train and Val folders
            for split in ['Train', 'Val']:
                split_path = os.path.join(class_path, split)
                if os.path.exists(split_path):
                    print(f"Processing {split} folder for class: {class_name}")

                    # Loop through categories: Defect_mask, Degraded_image, and clean_image
                    for category, folder_name in zip(
                        ['Defect_mask', 'Degraded_image', 'GT_clean_image'],
                        ['defect_mask','degraded', 'ground_truth']
                    ):
                        category_path = os.path.join(split_path, category)
                        if os.path.exists(category_path):
                            # Loop through defect types (subclass)
                            for defect_type in os.listdir(category_path):
                                defect_type_path = os.path.join(category_path, defect_type)

                                if os.path.isdir(defect_type_path):
                                    print(f"  Processing defect type: {defect_type} in category: {category}")

                                    # Define the destination directory
                                    dest_category_path = os.path.join(dest_dir, split, folder_name)
                                    os.makedirs(dest_category_path, exist_ok=True)

                                    # Copy images with renamed filenames
                                    for i, filename in enumerate(sorted(os.listdir(defect_type_path)), start=1):
                                        src_image_path = os.path.join(defect_type_path, filename)
                                        new_filename = f"{class_name}_{defect_type}_{i:03d}.png"
                                        dest_image_path = os.path.join(dest_category_path, new_filename)
                                        
                                        if os.path.isfile(src_image_path):
                                            shutil.copy2(src_image_path, dest_image_path)
                                            print(f"Copied {src_image_path} to {dest_image_path}")
                                        else:
                                            print(f"Skipped non-file item: {src_image_path}")
                        else:
                            print(f"Category {category} does not exist in {split} for class {class_name}")

## USER INPUT
# Rearrange files in the dataset
src_dir = 'Dataset/Denoising_Dataset_train_val'  # Source directory of the original dataset structure
dest_dir = 'Dataset/structured_data'     # Destination directory for the rearranged structure
rearrange_and_rename_files(src_dir, dest_dir)

# Continue with model loading and evaluation as in your code
file_id = '1gGiza9UsHM679TDlvn-1fhhu6V2Y0hS2'
url = f'https://drive.google.com/uc?id={file_id}'
checkpoint_path = 'Model/checkpoint_epoch_18.pth'

if not os.path.exists(checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    gdown.download(url, checkpoint_path, quiet=False)

model = load_model(checkpoint_path, encoder_name='resnet34')
input_folder = 'Dataset/structured_data/Val/degraded'
ground_truth_folder = 'Dataset/structured_data/Val/ground_truth'
mask_folder = 'Dataset/structured_data/Val/defect_mask'
save_output_folder = 'output_results'

# Run on the first n images, e.g., n=5
infer_and_evaluate(model, input_folder, ground_truth_folder, mask_folder, save_output_folder, n_images=5)

