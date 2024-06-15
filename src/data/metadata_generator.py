import os
import csv

# Define the base directory
base_dir = "data/preprocessed"

# Paths for experiment and holding images and masks
experiment_path = os.path.join(base_dir, "experiment")
holding_images_path = os.path.join(base_dir, "holding/holding_images")
holding_masks_path = os.path.join(base_dir, "holding/holding_masks")

# Initialize list to hold the metadata
metadata = []

# Function to collect metadata from a given directory
def collect_metadata(split, images_dir, masks_dir, relative_image_prefix, relative_mask_prefix):
    for image_file in os.listdir(images_dir):
        if image_file.endswith("_image.png"):
            image_id = image_file.split('_')[0]
            image_path = os.path.join(relative_image_prefix, image_file).replace('\\', '/')
            mask_file = f"{image_id}_mask.png"
            mask_path = os.path.join(relative_mask_prefix, mask_file).replace('\\', '/')
            metadata.append({
                "image_id": image_id,
                "split": split,
                "image_path": image_path,
                "mask_path": mask_path
            })

# Collect metadata from experiment split
collect_metadata("experiment", experiment_path, experiment_path,
                 "experiment", "experiment")

# Collect metadata from holding split
collect_metadata("holding", holding_images_path, holding_masks_path,
                 "holding/holding_images", "holding/holding_masks")

# Path to the metadata.csv file
metadata_csv_path = os.path.join(base_dir, "metadata.csv")

# Write metadata to csv file
with open(metadata_csv_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["image_id", "split", "image_path", "mask_path"])
    writer.writeheader()
    for data in metadata:
        writer.writerow(data)

print(f"metadata.csv has been created at {metadata_csv_path}")
