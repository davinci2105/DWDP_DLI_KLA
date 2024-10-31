
## Folder and File Descriptions

### Data/
Contains scripts for data preparation and preprocessing. This includes cleaning, inspecting, and preparing the dataset.

- **`__init__.py`**: Initializes the `Data` module.
- **`clean_dataset.py`**: Script to clean raw data, removing any corrupt or irrelevant files.
- **`inspect_dataset.py`**: Script for inspecting data characteristics, such as image dimensions and pixel distributions.
- **`prepare.py`**: Prepares the dataset by organizing it according to the model's requirements.
- **`workflow.ipynb`**: Jupyter notebook that provides an interactive workflow for data processing.

### Experiments/
This directory holds the scripts and notebooks necessary for model experimentation and training.

- **`__init__.py`**: Initializes the `Experiments` module.
- **`Attention_based_unet.ipynb`**: Notebook for experimenting with the Attention U-Net architecture, used in denoising tasks.
- **`dataset.py`**: Manages dataset loading and processing for model training and evaluation.
- **`infer_gui.py`**: Script that provides a GUI interface for performing inference using the trained model.
- **`infer_simple.py`**: Basic script for running inference without a GUI.
- **`model.py`**: Contains the model architecture, defining layers and structures used in the denoising network.
- **`test.py`**: Script to test the model's performance on test data, providing metrics for evaluation.
- **`train.py`**: Training script to train the model on the dataset.
- **`utils.py`**: Utility functions used across training, testing, and data processing tasks.

### Model/
Stores model checkpoints, which are saved weights of the model after each epoch or specified training intervals. These can be used to resume training or perform inference.

- **`checkpoint_epoch_xx.pth`**: Saved model checkpoint for epoch `xx`. Multiple checkpoints may be stored for different epochs to track model progress.

### Other Files
- **`.gitignore`**: Specifies files and folders that should be ignored by Git, such as large data files or sensitive information.
- **`Denoising_Dataset_train_val.zip`**: Zipped dataset containing training and validation data. It should be extracted before use.
- **`README.md`**: This README file, providing an overview of the project structure and purpose of each component.

## Usage

1. **Data Preparation**:
   - Use scripts in the `Data/` folder to clean, inspect, and prepare your dataset.
   - For an interactive workflow, open `workflow.ipynb`.

2. **Model Training**:
   - Navigate to `Experiments/` and use `train.py` to start training the model.
   - You can adjust model architecture in `model.py` and dataset handling in `dataset.py`.

3. **Inference**:
   - After training, use `infer_simple.py` for quick inference or `infer_gui.py` for a GUI-based inference experience.
   - Model checkpoints from `Model/` can be loaded for inference.

4. **Testing and Evaluation**:
   - Use `test.py` to evaluate model performance on test data. Metrics can be adjusted and computed as needed.

## Requirements

To run this project, you will need the following Python libraries (listed in `requirements.txt` if available):

```bash
pip install -r requirements.txt
