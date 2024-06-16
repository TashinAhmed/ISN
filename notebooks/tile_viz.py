#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "15/05/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed   
# version      : "0.0.1"
# status       : "PoC"
# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def load_images_from_dir(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        img = mpimg.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
    return images

def load_images_and_masks(images_dir, masks_dir):
    images = load_images_from_dir(images_dir)
    masks = load_images_from_dir(masks_dir)
    return images, masks

def plot_and_save_combined_images(images, masks, save_filename):
    num_images = min(len(images), 10)  
    fig, axs = plt.subplots(2, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        axs[0, i].imshow(images[i])
        axs[0, i].axis('off')
        axs[0, i].set_title(f'Image {i+1}')

        axs[1, i].imshow(masks[i])
        axs[1, i].axis('off')
        axs[1, i].set_title(f'Mask {i+1}')

    plt.tight_layout()
    plt.savefig(save_filename)
    plt.show()

images_dir = 'data/images'
masks_dir = 'data/masks'

images, masks = load_images_and_masks(images_dir, masks_dir)

save_filename = 'assets/dataset'
plot_and_save_combined_images(images, masks, save_filename)
