import os
import shutil

# Define the paths for the source and destination
source_dir = "Dataset/structured_data_11"  # Replace with your actual path
destination_dir = "Dataset/new_data"       # Replace with your desired destination path

# Paths for degraded and ground truth
degraded_path = os.path.join(source_dir, "Val", "degraded")
ground_truth_path = os.path.join(source_dir, "Val", "ground_truth")

# New paths for noise and clean
noise_path = os.path.join(destination_dir, "Noise")
clean_path = os.path.join(destination_dir, "Clean")

# Create the destination directory and subdirectories if they don't exist
os.makedirs(destination_dir, exist_ok=True)
os.makedirs(noise_path, exist_ok=True)
os.makedirs(clean_path, exist_ok=True)

# Copy files from degraded to noise folder
for file_name in os.listdir(degraded_path):
    source_file = os.path.join(degraded_path, file_name)
    dest_file = os.path.join(noise_path, file_name)
    shutil.copy2(source_file, dest_file)

# Copy files from ground truth to clean folder
for file_name in os.listdir(ground_truth_path):
    source_file = os.path.join(ground_truth_path, file_name)
    dest_file = os.path.join(clean_path, file_name)
    shutil.copy2(source_file, dest_file)

print("Files copied successfully!")
