
### Rearranged Dataset Structure

After running the `rearrange_and_rename_files` function, the structure will be modified to:

Dataset/ └── structured_data/ ├── Train/ │ ├── defect_mask/ │ ├── degraded/ │ └── ground_truth/ └── Val/ ├── defect_mask/ ├── degraded/ └── ground_truth/


This rearranged structure organizes images by data split (`Train`/`Val`) and category, making them ready for model processing.

---

## Usage

1. **Rearrange Dataset Files**: Use the `rearrange_and_rename_files` function to structure the dataset as required. This will organize and rename files within the dataset.

    ```python
    src_dir = 'Dataset/Denoising_Dataset_train_val'
    dest_dir = 'Dataset/structured_data'
    rearrange_and_rename_files(src_dir, dest_dir)
    ```

2. **Download Model Checkpoint**: The code checks if the model checkpoint exists locally. If not, it will download it from Google Drive.

3. **Run Inference and Evaluation**: Use the `infer_and_evaluate` function to process the images, calculate metrics, and create a report.

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

## Model Details

This project uses an attention UNet model with `scSE` (spatial and channel squeeze and excitation) attention implemented via `segmentation_models_pytorch`. Model features include:

- **Attention Mechanism**: The scSE attention block enhances segmentation and defect preservation.
- **Configurable Encoder**: The encoder type can be changed in the `load_model()` function (`resnet34` is the default).
- **Pretrained Weights**: By default, the model uses pretrained weights for the encoder, set to `imagenet`.

---

## Troubleshooting

- **Checkpoint Download Issues**: Ensure you have an active internet connection if the model checkpoint needs downloading.
- **Missing Folders or Files**: Ensure the dataset folders are named correctly and have the required format before running the rearrangement function.
- **Dependencies**: Ensure all dependencies from `requirements.txt` are installed, and `torch` and `torchvision` are compatible with your system’s CUDA setup for GPU support.
- **Metric Calculation Errors**: If you encounter errors with metric calculations, ensure all input images are normalized to a `[0, 1]` range.

---

## License

This project is licensed under the MIT License.
