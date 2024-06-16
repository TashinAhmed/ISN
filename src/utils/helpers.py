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

"""
Helper functions for handling semantic segmentation tasks using numpy arrays.

This script contains utility functions for:
1. Reversing one-hot encoded segmentation masks to class indices.
2. Colour coding segmentation masks based on specified label values.

Imports:
- numpy as np: Importing numpy for numerical operations.

Functions:
- reverse_one_hot(image): Converts a one-hot encoded image into class indices.
- colour_code_segmentation(image, label_values): Colour codes segmentation mask based on label values.
- main(): Entry point of the script, prints a message indicating that helper functions are ready.

Usage:
- Run this script to use the helper functions for semantic segmentation tasks.

Exceptions:
- None handled directly within the script, functions assume input arrays are correctly formatted.

"""

import numpy as np


def reverse_one_hot(image):
    """
    Reverse one-hot encoded segmentation mask to class indices.

    Args:
    - image (np.ndarray): One-hot encoded segmentation mask of shape (C, H, W).

    Returns:
    - np.ndarray: Segmentation mask with class indices of shape (H, W).
    """
    x = np.argmax(image, axis=0)
    return x


def colour_code_segmentation(image, label_values):
    """
    Colour code segmentation mask based on specified label values.

    Args:
    - image (np.ndarray): Segmentation mask of shape (H, W).
    - label_values (list): List of RGB values corresponding to each class label.

    Returns:
    - np.ndarray: Colour coded segmentation mask with shape (H, W, 3).
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x


def main():
    print("Helper functions are ready to use.")


if __name__ == "__main__":
    main()
