#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "16/06/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed   
# version      : "0.0.1"
# status       : INCOMPLETE
# ----------------------------------------------------------------------------

"""
Utility function for visualizing images using matplotlib.

This script contains a utility function to visualize images:
- visualize(**images): Plot images in a single row with their respective titles.

Imports:
- matplotlib.pyplot as plt: Importing matplotlib for plotting images.

Functions:
- visualize(**images): Plots images in a single row with titles derived from keys in the **images dictionary.
- main(): Entry point of the script, prints a message indicating that visualization utilities are ready.

Usage:
- Run this script to use the visualize function for plotting images.

Exceptions:
- Assumes input images are provided as dictionaries with keys as image names and values as image arrays.

"""

import matplotlib.pyplot as plt


def visualize(**images):
    """
    Plot images in one row with titles.

    Args:
    - **images (dict): Dictionary where keys are image names and values are image arrays (np.ndarray).
      Each image will be plotted with its title derived from the key.

    Returns:
    - None
    """
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


def main():
    print("Visualization utilities are ready to use.")


if __name__ == "__main__":
    main()
