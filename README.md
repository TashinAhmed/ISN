# ISN

Just another PoC on Image Segmentation Network on dummy dataset

### Directory Tree

```
.
└── ISN/
    ├── assets/
    │   └── PRESENTABLE_ASSETS
    ├── data/
    │   ├── preprocessed/
    │   │   ├── experiment
    │   │   ├── holding/
    │   │   │   ├── holding_images
    │   │   │   └── holding_masks
    │   │   ├── class_dict.csv
    │   │   └── metadata.csv
    │   └── raw/
    │       ├── images
    │       └── masks
    ├── docs/
    │   └── PRIVATE_DOCS
    ├── experiments/
    │   └── PERFORMED_EXPERIMENTS
    ├── notebooks/
    │   ├── dataset_viz.ipynb
    │   └── trial_isn_segmentation.ipynb
    ├── scripts/
    │   ├── __init__.py
    │   ├── run_all.sh
    │   ├── run_evaluation.py
    │   ├── run_training.py
    │   └── visualize_results.py
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── constants.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── data_preprocessing.py
    │   │   ├── datamodule.py
    │   │   └── dataset.py
    │   ├── data_preparation/
    │   │   ├── __init__.py
    │   │   ├── dataset_generator.py
    │   │   ├── dataset_preprocessing.py
    │   │   └── metadata_generator.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── segmentation_model.py
    │   │   └── unet_model.py
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── test.py
    │   │   └── train.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── checkpoint_utils.py
    │       ├── helpers.py
    │       ├── metrics.py
    │       └── visualization.py
    ├── working/
    │   ├── lightning_logs
    │   ├── outputs
    │   └── savedckpt
    ├── .gitignore
    ├── personal_journal.md
    ├── README.md
    └── requirements.txt
```

### File Structure Details

| File/Directory                                      | Description                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| assets                                              | data visuals and presentation related files                                                                                                                                                                                                                                                                                                                                                                    |
| data/preprocessed/<br />experiment                  | 900+900 image and mask (GT) data for train-test-eval<br />at ratio 8:1:1                                                                                                                                                                                                                                                                                                                                       |
| data/preprocessed/<br />holding                     | two directories under this dir. 1 for images and 1 for GT masks.<br />100 data per dir. This holding set is for final test before shipping the model.                                                                                                                                                                                                                                                        |
| data/preprocessed/<br />class_dict.csv              | consists the ROI color code. For target rectangle (255,255,255) and rest<br />are background (0,0,0)                                                                                                                                                                                                                                                                                                           |
| data/preprocessed/<br />metadata.csv                | assisting metadata for the dataset, for faster navigation and data processing.<br />consists of ID, image split names and relative paths for images and masks.                                                                                                                                                                                                                                                |
| data/raw/                                           | consists of raw images and masks generated by ``src/dataset_preparation/dataset_generator.py``                                                                                                                                                                                                                                                                                                                 |
| docs                                                | Some private PDF files passed upon, consists of problem statement and so on.                                                                                                                                                                                                                                                                                                                                   |
| experiments                                         | some rough scripts to do quick tests and their corresponding outcomes.<br />***It don't consist any unit tests or organized experiments.***                                                                                                                                                                                                                                                            |
| notebooks/dataset_viz.ipynb                         | a simple trial on data visualization                                                                                                                                                                                                                                                                                                                                                                           |
| notebooks/tile_viz.py                               | same scripts as ``notebooks/dataset_viz.ipynb`` but in .py format.                                                                                                                                                                                                                                                                                                                                             |
| notebooks/<br />trial_isn_segmentation.ipynb        | primary codebase/notebook for end to end training code that works. As I have worked on<br />colab notebook initially. The notebook is the primary experimentation place. After the initial<br />development under this notebook for train-test-eval, I have moved the code scripts in a packaged file format.                                                                                                 |
| scripts/run_all.sh                                  | run all the .py scripts under ``scripts`` one by one, starting from ``run_training.py``, ``run_evaluation.py``, ``visualize_results.py``. Each script will run after a min time interval.<br />Caution: `visualize_results.py` has some error which have not been handled properly, thus it might throw an error.                                                                                           |
| scripts/run_evaluation.py                           | trigger the ``src/training/test.py``                                                                                                                                                                                                                                                                                                                                                                           |
| scripts/run_training.py                             | trigger the ``src/training/train.py``                                                                                                                                                                                                                                                                                                                                                                          |
| scripts/visualize_results.py                        | trigger the ``src/utils/visualization.py``                                                                                                                                                                                                                                                                                                                                                                     |
| src/data/<br />data_preprocessing.py                | Utility functions to load metadata, split data, and load class information from CSV files.<br />Split data related function is a little bit of unorthodox as I was thinking to do train-eval only. After that,<br />pivoted to train-test-val. Should have used sklearn to do the split in one line, but the multi-liner works too.                                                                            |
| src/data/datamodule.py                              | This module handles loading and preprocessing of ISN dataset for train-test-val using Pytorch Lighning Dataloader.                                                                                                                                                                                                                                                                                             |
| src/data/dataset.py                                 | loads images and masks from provided dataframe, applies image augmentations liek different transformations<br />i.e. preprocess images.                                                                                                                                                                                                                                                                        |
| src/data_preparation/<br />dataset_generator.py     | Generates random shape images and masks based on provided configurations. In short, generates data for the whole experiment<br />and they are saved in `data/raw/images` and ``data/raw/masks.``                                                                                                                                                                                                             |
| src/data_preparation/<br />dataset_preprocessing.py | This module preprocess the `data/raw/` dataset by splitting it into `experiment` and `holding` sets.                                                                                                                                                                                                                                                                                                     |
| src/data_preparation/<br />metadata_generator.py    | Script to collect metadata for experiment and holding datasets and save it to a CSV file named `metadata.csv`                                                                                                                                                                                                                                                                                                |
| src/models/<br />segmentation_model.py              | PyTorch Lightning Segmentation Model for training, validation, and testing. This module defines a segmentation model using<br />segmentation_models_pytorch, incorporating metrics such as IoU score, F1 score, acc and recall for eval during train, test and val.                                                                                                                                            |
| src/models/<br />unet_model.py                      | Utilized UNet++ model using segmentation_models_pytorch library.                                                                                                                                                                                                                                                                                                                                               |
| src/training/train.py                               | Training script for training a segmentation model using PyTorch Lightning. This script loads metadata and prepares training, validation,<br />and test datasets. Creates a UNet model using segmentation_models_pytorch library. Defines loss function, early stopping callback, <br />checkpoint callback, and logger. Initializes PyTorch Lightning Trainer and performs training of the segmentation model. |
| src/training/test.py                                | Test script for evaluating a segmentation model using a test dataset and a saved checkpoint. This script loads metadata and prepares<br />test dataset. Creates a UNet model using segmentation_models_pytorch library. Loads the latest checkpoint of a SegmentationModel. <br />Performs testing of the loaded model on the test dataset.                                                                    |
| src/utils/checkpoint_utils.py                       | to find latest .ckpt (weight file) from designated dir and update the new trained weight with new version name.                                                                                                                                                                                                                                                                                                |
| src/utils/helpers.py                                | Helper functions for handling semantic segmentation tasks using numpy arrays. This script contains utility functions for reversing one-hot<br />encoded segmentation masks to class indices. Colour coding segmentation masks based on specified label values.                                                                                                                                                 |
| src/utils/metrics.py                                | This script contains a utility function to compute segmentation metrics IOU (Intersection over Union) score, F1 score, Accuracy, Recall                                                                                                                                                                                                                                                                        |
| src/utils/visualization.py                          | Utility function for visualizing images using matplotlib.                                                                                                                                                                                                                                                                                                                                                      |
| src/config.py                                       | all the hyperparameters init, can be used as an .yaml file too.                                                                                                                                                                                                                                                                                                                                                |
| src/constants.py                                    | hyperparameters/ vars that needs more attention before altering.                                                                                                                                                                                                                                                                                                                                               |
| working/lightning_logs                              | detailed hyperparameters under `version_#/hparams.yaml` and `metrics.csv` has the outcome of all the metrics for train-test-eval.                                                                                                                                                                                                                                                                         |
| working/outputs                                     | some image outputs from the 10% test data from `data/experiment`.                                                                                                                                                                                                                                                                                                                                           |
| working/savedckpt                                   | weight files of every successful train-test-eval phases                                                                                                                                                                                                                                                                                                                                                        |
| personal_journal.md                                 | my personal scrapbook while working in this PoC                                                                                                                                                                                                                                                                                                                                                                |
| requirements.txt                                    | required python packages and version names utilized in this PoC. Can be altered with poetry, yaml and others as well.                                                                                                                                                                                                                                                                                          |

### Instructions to Run

1. Run `python src/data_preparation/dataset_generator.py` from the parent directory. The script will generate data (images and masks) under the parent directory named as `data`. argparser also available. `python src/data_preparation/dataset_generator.py --img_height 256 --img_width 256 --min_shape_px 30 --max_shape_px 50 --num_images 100 --image_dir 'data/raw/images' --mask_dir 'data/raw/masks'`
2. After that, run `python src/data_preparation/dataset_prepocessing.py` (argparse available for this script as well). This script will process raw data and create training ready data directories.
3. Run python `src/data_preparation/metadata_generator.py` which will generate `metadata.csv`.
4. `data/preprocessed/class_dict.csv` have to create manually for now according to the format.
5. To train run `python scripts/run_training.py`
6. To eval run `python scripts/run_evaluation.py`
7. Alternatively run `./run_all.sh` from `scripts` directory. (`visualize_results.py` might throw some error at this point)
8. To generate logs run `python scripts/run_{training/evaluation}.py > logs/train.log 2>&1`


### Branch Information

* main: final branch
* datagen: on data generation processing
* train: train-test-eval experimentation
