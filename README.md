# ISN

Just another PoC on Image Segmentation Network on dummy dataset

### Instructions:

Run ``python dataset_generator/dataset_generator.py``(9.86 rating on pylint) from the parent directory. The script will generate data (images and masks) under the parent directory named as ``data``. Structure of the ``data`` directory is,

```
ISN/
├── data/
│   ├── images/
│   │   ├── 1_image.png
│   │   ├── 2_image.png
│   │   └── ...
│   └── masks/
│       ├── 1_mask.png
│       ├── 2_mask.png
│       └── ...
```

argparser available thus the default arguments can be updated: ``python generate_shapes.py --img_height 256 --img_width 256 --min_shape_px 30 --max_shape_px 50 --num_images 100 --image_dir 'output/images' --mask_dir 'data/masks'``
