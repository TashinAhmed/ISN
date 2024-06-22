#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "23/06/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "PoC"
# ----------------------------------------------------------------------------

"""
Test script for evaluating a segmentation model using a test dataset and a saved checkpoint.

This script:
1. Loads metadata and prepares test dataset.
2. Creates a UNet model using segmentation_models_pytorch library.
3. Loads the latest checkpoint of a SegmentationModel.
4. Performs testing of the loaded model on the test dataset.

Imports:
- os: Provides functionalities for interacting with the operating system.
- segmentation_models_pytorch as smp: Imports the segmentation model library.
- torch.nn: Imports neural network modules from PyTorch.
- pytorch_lightning as pl: Imports PyTorch Lightning for training and testing.
- src.models.unet_model.create_unet_model: Imports function to create a UNet model.
- src.models.segmentation_model.SegmentationModel: Imports SegmentationModel class for loading checkpoints.
- src.data.datamodule.ISNDataModule: Imports data module for preparing dataset.
- src.data.data_preprocessing: Imports functions for loading metadata and class information.
- src.config: Imports configuration variables for the experiment.
- src.utils.checkpoint_utils.get_latest_checkpoint: Imports function to get the latest checkpoint file.

Functions:
- test(): Performs testing of the segmentation model.
- main(): Entry point of the script.

Usage:
- Run this script to evaluate a segmentation model on the test dataset using the latest checkpoint.

Exceptions:
- FileNotFoundError: Raised if no checkpoint file is found in the specified directory.
"""

import os

import segmentation_models_pytorch as smp
from torch import nn

import pytorch_lightning as pl
from src.models.unet_model import create_unet_model
from src.models.segmentation_model import SegmentationModel
from src.data.datamodule import ISNDataModule
from src.data.data_preprocessing import load_metadata, split_data, load_class_info
from src.config import *
from src.utils.checkpoint_utils import get_next_checkpoint_filename


def test():
    metadata_df = load_metadata(DATA_DIR)
    FOO, BAR, test_df = split_data(metadata_df)
    class_names, class_rgb_values = load_class_info(DATA_DIR)

    net = create_unet_model()
    loss = nn.CrossEntropyLoss()
    preprocess_input = smp.encoders.get_preprocessing_fn(
        ENCODER, pretrained=ENCODER_WEIGHTS
    )
    isn_data = ISNDataModule(
        train_df=FOO,
        valid_df=BAR,
        test_df=test_df,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        preprocess_fn=preprocess_input,
        class_rgb_values=class_rgb_values,
    )

    # segmodel = SegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # latest_checkpoint = get_next_checkpoint_filename(CHECKPOINT_DIR, "lightning_trained")
    # if latest_checkpoint:
    #     checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    #     print(f"Loading checkpoint: {checkpoint_path}")
    #     segmodel = SegmentationModel.load_from_checkpoint(
    #         checkpoint_path=checkpoint_path
    #     )
    # else:
    #     raise FileNotFoundError("No checkpoint file found.")

    next_checkpoint, latest_checkpoint = get_next_checkpoint_filename(
        CHECKPOINT_DIR, "lightning_trained"
    )

    if latest_checkpoint:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
        print(f"Loading checkpoint: {checkpoint_path}")
        segmodel = SegmentationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
    else:
        raise FileNotFoundError("No checkpoint file found.")

    next_checkpoint_path = os.path.join(CHECKPOINT_DIR, next_checkpoint)
    if os.path.exists(next_checkpoint_path):
        print(f"Loading next checkpoint: {next_checkpoint_path}")
        segmodel = SegmentationModel.load_from_checkpoint(
            checkpoint_path=next_checkpoint_path
        )
    else:
        print(f"Next checkpoint not found. Falling back to: {checkpoint_path}")

    trainer = pl.Trainer(accelerator=DEVICE, devices=NUM_DEVICES)
    trainer.test(segmodel, datamodule=isn_data)


def main():
    test()


if __name__ == "__main__":
    main()
