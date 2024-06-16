#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "16/06/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed   
# version      : "0.0.1"
# status       : "PoC"
# ----------------------------------------------------------------------------

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def one_hot_encode(label, label_values):
    """
    Perform one-hot encoding on the label using given label values.

    Args:
    - label (np.array): RGB image array representing the label.
    - label_values (list): List of RGB tuples for each class.

    Returns:
    - np.array: One-hot encoded semantic map.
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


class ISNSet(Dataset):
    """
    Dataset class for ISN dataset.

    This class loads images and corresponding masks from provided DataFrame,
    applies transformations, and optionally preprocesses images.

    Initializes ISNSet with dataset information and transformations.

    Args:
    - df (pd.DataFrame): DataFrame containing image and mask file paths.
    - transform (callable): Optional transformations to apply to images and masks.
    - preprocess_fn (callable): Optional preprocessing function for input images.
    - class_rgb_values (list): List of RGB tuples for each class in the dataset.
    """
    def __init__(self, df, transform=None, preprocess_fn=None, class_rgb_values=None):
        self.image_paths = df["image_path"].tolist()
        self.mask_paths = df["mask_path"].tolist()
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.class_rgb_values = class_rgb_values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask for a given index.

        Args:
        - idx (int): Index to retrieve image and mask.

        Returns:
        - tuple: Tuple containing the processed image and its corresponding mask.
        """
        img_file = self.image_paths[idx]
        mask_file = self.mask_paths[idx]

        img = Image.open(img_file).convert("RGB")
        img = np.array(img)

        mask = Image.open(mask_file).convert("RGB")
        mask = one_hot_encode(mask, self.class_rgb_values).astype("float")

        if self.preprocess_fn:
            img = self.preprocess_fn(img)
            img = np.array(img, dtype=np.float32)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask


def main():
    print("Dataset class is ready to use.")


if __name__ == "__main__":
    main()
