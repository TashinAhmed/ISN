import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# Function to load images from a directory
def load_images_from_dir(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        img = mpimg.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)
    return images

# Function to load images and masks from their respective directories
def load_images_and_masks(images_dir, masks_dir):
    images = load_images_from_dir(images_dir)
    masks = load_images_from_dir(masks_dir)
    return images, masks

# Function to plot images and masks in rows and save as one figure
def plot_and_save_combined_images(images, masks, save_filename):
    num_images = min(len(images), 10)  # Limit to first 10 images
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

# Directory paths where images and masks are stored
images_dir = 'data/images'
masks_dir = 'data/masks'

# Load images and masks
images, masks = load_images_and_masks(images_dir, masks_dir)

# Filename to save combined image and masks
save_filename = 'assets/dataset'

# Display and save the first 10 images and their masks as one figure
plot_and_save_combined_images(images, masks, save_filename)
