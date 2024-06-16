#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "15/06/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed   
# version      : "0.0.1"
# status       : "PoC"
# ----------------------------------------------------------------------------

"""
Script to collect metadata for experiment and holding datasets and save it to a CSV file named metadata.csv
"""

import os
import csv

base_dir = "data/preprocessed"

experiment_path = os.path.join(base_dir, "experiment")
holding_images_path = os.path.join(base_dir, "holding/holding_images")
holding_masks_path = os.path.join(base_dir, "holding/holding_masks")

metadata = []


def collect_metadata(
    split, images_dir, masks_dir, relative_image_prefix, relative_mask_prefix
):
    """
    Collects metadata for images and masks in the specified directories.

    Args:
    - split (str): The split type (e.g., "experiment" or "holding").
    - images_dir (str): Directory path containing images.
    - masks_dir (str): Directory path containing masks.
    - relative_image_prefix (str): Relative prefix for image paths in metadata.
    - relative_mask_prefix (str): Relative prefix for mask paths in metadata.
    """
    for image_file in os.listdir(images_dir):
        if image_file.endswith("_image.png"):
            image_id = image_file.split("_")[0]
            image_path = os.path.join(relative_image_prefix, image_file).replace(
                "\\", "/"
            )
            mask_file = f"{image_id}_mask.png"
            mask_path = os.path.join(relative_mask_prefix, mask_file).replace("\\", "/")
            metadata.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )


collect_metadata(
    "experiment", experiment_path, experiment_path, "experiment", "experiment"
)

collect_metadata(
    "holding",
    holding_images_path,
    holding_masks_path,
    "holding/holding_images",
    "holding/holding_masks",
)

metadata_csv_path = os.path.join(base_dir, "metadata.csv")


with open(metadata_csv_path, mode="w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=["image_id", "split", "image_path", "mask_path"]
    )
    writer.writeheader()
    for data in metadata:
        writer.writerow(data)

print(f"metadata.csv has been created at {metadata_csv_path}")
