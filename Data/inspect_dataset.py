import os
from collections import defaultdict
from PIL import Image

# Define the base directory
base_dir = '/root/DLI_KLA/structured_data'
categories = ['train', 'val']
folders = ['degraded', 'defect_mask', 'ground_truth']

# Function to count and print image shapes
def count_image_shapes(category):
    shape_counts = defaultdict(int)  # Dictionary to store shape counts
    
    # Iterate through each folder (degraded, defect_mask, ground_truth)
    for folder in folders:
        folder_path = os.path.join(base_dir, category, folder)
        
        print(f"\nCounting shapes in folder: {folder} ({category})")
        
        # Loop through each image in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            try:
                # Open image using PIL
                with Image.open(image_path) as img:
                    width, height = img.size
                    shape = (height, width)  # Create tuple for shape (H x W)
                    shape_counts[shape] += 1  # Increment count for this shape
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
    
    # Print the shape counts
    print(f"\nImage shape counts in '{category}' category:")
    for shape, count in shape_counts.items():
        print(f"Shape (H x W): {shape[0]} x {shape[1]} - {count} images")

# Run the function for both train and val categories
for category in categories:
    count_image_shapes(category)

print("\nImage shape counting completed.")
