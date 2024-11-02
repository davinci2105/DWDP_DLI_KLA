# Image Denoising with Defect Preservation using Attention U-Net

This project uses a deep learning model, specifically an Attention U-Net, to perform image denoising while preserving defect regions. The model is trained and evaluated on a dataset containing degraded images, defect masks, and clean ground truth images. Key metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure) are used to assess the quality of the denoised images, with results compiled in a PDF report.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Training Setup](#training-setup)
- [Testing Setup](#testing-setup)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Reference Papers](#reference-papers)

---

## Overview

The aim of this project is to enhance image quality by denoising images while retaining important defect regions. We employ an Attention U-Net model with attention mechanisms to better focus on defect areas, ensuring they are preserved through the denoising process. This is particularly useful for applications where the accuracy of defect detection is crucial, such as quality inspection or medical imaging.

---

## Architecture

### Attention U-Net Model

This project uses an Attention U-Net model built using the `segmentation_models_pytorch` library. The U-Net model is augmented with attention mechanisms that help the model focus on specific areas (defects) during training and inference. The following are key elements of the architecture:

1. **Encoder**: A ResNet34 backbone is used as the encoder to extract feature representations from input images. This backbone can be changed if needed by modifying the `encoder_name` parameter.
  
2. **Attention Mechanism**: An scSE (spatial and channel Squeeze and Excitation) block is incorporated to apply spatial and channel-based attention. This helps the model prioritize certain regions (e.g., defects) in the images, aiding in defect preservation during denoising.

3. **Decoder**: The decoder upsamples the feature maps to reconstruct a denoised version of the original image while maintaining the resolution of the defects.

4. **Skip Connections**: Similar to traditional U-Net architecture, skip connections are used between the encoder and decoder to preserve spatial details in the reconstruction.

### Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Evaluates image quality by comparing pixel intensities between the output and ground truth.
- **SSIM (Structural Similarity Index Measure)**: Measures structural similarity between images, with higher scores indicating better similarity.

These metrics are calculated for both the whole image and masked regions to ensure defect preservation.

---

## Installation

To set up the project:

1. Clone the repository and navigate to the directory:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that your environment has compatible versions of `torch` and `torchvision` for GPU (CUDA) support if available.

---

## Dataset Structure

### Input Dataset Structure

The dataset should be structured as follows **before** running the rearrangement function:

Dataset/ └── Denoising_Dataset_train_val/ ├── Class1/ │ ├── Train/ │ │ ├── Defect_mask/ │ │ ├── Degraded_image/ │ │ └── GT_clean_image/ │ └── Val/ │ ├── Defect_mask/ │ ├── Degraded_image/ │ └── GT_clean_image/ └── Class2/ ├── Train/ └── Val/


### Rearranged Dataset Structure

After running the `rearrange_and_rename_files` function, the dataset will have the following structure:

Dataset/ └── structured_data/ ├── Train/ │ ├── defect_mask/ │ ├── degraded/ │ └── ground_truth/ └── Val/ ├── defect_mask/ ├── degraded/ └── ground_truth/


This rearranged structure organizes images by data split (`Train`/`Val`) and category, making them ready for model processing.

---

## Training Setup

The project assumes the model has already been trained and saved in the form of a checkpoint file. If training is required, consider implementing a training loop with loss functions suited to image denoising and defect preservation. You could use the following loss functions:

1. **Mean Squared Error (MSE)**: Commonly used for pixel-wise comparison.
2. **Structural Similarity Index Loss**: Ensures structural similarity, crucial for defect preservation.

The checkpoint, when saved, can then be loaded for evaluation using the script provided.

---

## Testing Setup

The testing setup involves using the pre-trained model to denoise images in the validation dataset and calculate evaluation metrics.

1. **Load Model Checkpoint**: If the model checkpoint does not exist, the code automatically downloads it from Google Drive.
2. **Inference and Evaluation**: During inference, the model processes each degraded image, applies the defect mask, and calculates metrics for both whole and masked regions.
3. **Output Report**: A PDF report is generated, including input images, ground truth, defect masks, denoised output, and calculated metrics.

---

## Usage

1. **Rearrange Dataset Files**: First, organize the dataset files using the `rearrange_and_rename_files` function:

    ```python
    src_dir = 'Dataset/Denoising_Dataset_train_val'
    dest_dir = 'Dataset/structured_data'
    rearrange_and_rename_files(src_dir, dest_dir)
    ```

2. **Download Model Checkpoint**: The code checks if the model checkpoint exists locally. If not, it will download it from Google Drive.

3. **Run Inference and Evaluation**: Use the `infer_and_evaluate` function to process images, calculate metrics, and generate a report.

   Example usage to process images and create a report:

    ```python
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
    ```

4. **View Results**: Check the `output_results/` directory for the generated PDF report and metric plots.

---

## Output

The output includes:

1. **PDF Report**: A report named `report.pdf` is generated in `output_results/<timestamp>/`, containing:
   - Input, ground truth, defect mask, and output images.
   - PSNR and SSIM metrics for both whole and masked regions.

2. **Average Metrics Plot**: `average_metrics.png`, which shows the mean PSNR and SSIM for each class, saved within the output folder.

---

## Troubleshooting

- **Checkpoint Download Issues**: Ensure you have an active internet connection if the model checkpoint needs downloading.
- **Missing Folders or Files**: Ensure the dataset folders are named correctly and have the required format before running the rearrangement function.
- **Dependencies**: Ensure all dependencies from `requirements.txt` are installed, and `torch` and `torchvision` are compatible with your system’s CUDA setup for GPU support.
- **Metric Calculation Errors**: If you encounter errors with metric calculations, ensure all input images are normalized to a `[0, 1]` range.

---

## Reference Papers

The following research papers were instrumental in guiding the architecture and approach of this project:

1. **Attention U-Net: Learning Where to Look for the Pancreas**  
   Olaf Ronneberger, Philipp Fischer, Thomas Brox  
   This paper introduced the concept of U-Net with attention mechanisms, providing insights on enhancing the model’s ability to focus on key areas in medical imaging.  
   *[Link](https://arxiv.org/abs/1804.03999)*

2. **Image Quality Assessment: From Error Visibility to Structural Similarity**  
   Zhou Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli  
   This foundational paper on SSIM provides the basis for evaluating structural similarity, a key metric used in this project.  
   *[Link](https://ieeexplore.ieee.org/document/1284395)*

3. **Squeeze-and-Excitation Networks**  
   Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Andrea Vedaldi  
   This paper presents the scSE (spatial and channel Squeeze and Excitation) mechanism, which inspired the attention mechanism applied to the U-Net architecture in this project.  
   *[Link](https://arxiv.org/abs/1709.01507)*

---

## License

This project is licensed under the MIT License.
