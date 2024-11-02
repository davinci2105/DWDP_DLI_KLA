import os

# Define base directory and categories
base_dir = 'structured_data'
categories = ['Train', 'Val']
folders = ['degraded', 'defect_mask', 'ground_truth']

# Function to get file names without extensions from a folder
def get_file_names(folder_path):
    return {os.path.splitext(f)[0] for f in os.listdir(folder_path)}

# Function to remove unmatched files and show deleted files
def remove_unmatched_files(category):
    # Paths to the folders
    degraded_path = os.path.join(base_dir, category, 'degraded')
    defect_mask_path = os.path.join(base_dir, category, 'defect_mask')
    ground_truth_path = os.path.join(base_dir, category, 'ground_truth')
    
    # Get file names (without extensions) in each folder
    degraded_files = get_file_names(degraded_path)
    defect_mask_files = get_file_names(defect_mask_path)
    ground_truth_files = get_file_names(ground_truth_path)
    
    # Find common files across all three folders
    common_files = degraded_files & defect_mask_files & ground_truth_files
    
    # Files to delete (those without a match in all three folders)
    print(f"\nUnmatched files to be deleted in '{category}' category:")

    # Remove unmatched files from degraded folder
    for file_name in degraded_files - common_files:
        file_to_remove = os.path.join(degraded_path, file_name + '.png')
        print(f"Deleting from degraded: {file_to_remove}")
        os.remove(file_to_remove)
    
    # Remove unmatched files from defect_mask folder
    for file_name in defect_mask_files - common_files:
        file_to_remove = os.path.join(defect_mask_path, file_name + '.png')
        print(f"Deleting from defect_mask: {file_to_remove}")
        os.remove(file_to_remove)
    
    # Remove unmatched files from ground_truth folder
    for file_name in ground_truth_files - common_files:
        file_to_remove = os.path.join(ground_truth_path, file_name + '.png')
        print(f"Deleting from ground_truth: {file_to_remove}")
        os.remove(file_to_remove)

# Run the function for both train and val categories
for category in categories:
    remove_unmatched_files(category)

print("\nUnmatched file deletion process completed successfully.")
