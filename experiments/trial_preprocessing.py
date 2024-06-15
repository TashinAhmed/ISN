import os
import shutil
from tqdm import tqdm


data_dir = 'data'  

experiment_dir = 'dataset/experiment'
holding_dir = 'dataset/holding'

holding_images_dir = os.path.join(holding_dir, 'holding_images')
holding_masks_dir = os.path.join(holding_dir, 'holding_masks')

os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(holding_images_dir, exist_ok=True)
os.makedirs(holding_masks_dir, exist_ok=True)

images_dir = os.path.join(data_dir, 'images')
masks_dir = os.path.join(data_dir, 'masks')

image_files = sorted(os.listdir(images_dir))
mask_files = sorted(os.listdir(masks_dir))

assert len(image_files) == len(mask_files), "Number of images and masks must be the same."

total_samples = len(image_files)
split_point = int(0.9 * total_samples)

for i in tqdm(range(total_samples)):
    prefix = os.path.splitext(image_files[i])[0].split('_')[0]  # Get prefix (e.g., '22')
    image_src = os.path.join(images_dir, image_files[i])
    mask_src = os.path.join(masks_dir, mask_files[i])
    
    if i < split_point:
        # Create experiment directory 
        image_dst = os.path.join(experiment_dir, f'{prefix}_image.png')
        mask_dst = os.path.join(experiment_dir, f'{prefix}_mask.png')
    else:
        # Create holding directory
        image_dst = os.path.join(holding_images_dir, f'{prefix}_image.png')
        mask_dst = os.path.join(holding_masks_dir, f'{prefix}_mask.png')
    
    shutil.copyfile(image_src, image_dst)
    shutil.copyfile(mask_src, mask_dst)

print(f"Dataset splitting and creation completed at\\{experiment_dir} and\\ {holding_dir}.")
