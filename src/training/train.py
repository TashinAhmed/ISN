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
Training script for training a segmentation model using PyTorch Lightning.

This script:
1. Loads metadata and prepares training, validation, and test datasets.
2. Creates a UNet model using segmentation_models_pytorch library.
3. Defines loss function, early stopping callback, checkpoint callback, and logger.
4. Initializes PyTorch Lightning Trainer and performs training of the segmentation model.

Imports:
- os: Provides functionalities for interacting with the operating system.
- pytorch_lightning as pl: Imports PyTorch Lightning for training and callbacks.
- pytorch_lightning.callbacks: Imports EarlyStopping and ModelCheckpoint callbacks.
- pytorch_lightning.loggers: Imports CSVLogger for logging training metrics.
- segmentation_models_pytorch as smp: Imports the segmentation model library.
- torch.nn: Imports neural network modules from PyTorch.
- src.models.unet_model.create_unet_model: Imports function to create a UNet model.
- src.models.segmentation_model.SegmentationModel: Imports SegmentationModel class for training.
- src.data.datamodule.ISNDataModule: Imports data module for preparing datasets.
- src.data.data_preprocessing: Imports functions for loading metadata and class information.
- src.config: Imports configuration variables for the experiment.
- src.utils.checkpoint_utils.get_next_checkpoint_filename: Imports function to generate next checkpoint filename.

Functions:
- train(): Performs training of the segmentation model.
- main(): Entry point of the script.

Usage:
- Run this script to train a segmentation model using PyTorch Lightning.

Exceptions:
- None handled directly within the script, but could raise exceptions related to file operations or data loading.
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import segmentation_models_pytorch as smp
from torch import nn

from src.models.unet_model import create_unet_model
from src.models.segmentation_model import SegmentationModel
from src.data.datamodule import ISNDataModule
from src.data.data_preprocessing import load_metadata, split_data, load_class_info
from src.config import *
from src.utils.checkpoint_utils import get_next_checkpoint_filename


def train():
    metadata_df = load_metadata(DATA_DIR)
    train_df, valid_df, test_df = split_data(metadata_df)
    class_names, class_rgb_values = load_class_info(DATA_DIR)

    net = create_unet_model()
    loss = nn.CrossEntropyLoss()

    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00001, patience=5, mode="min"
    )
    # checkpoint_callback = ModelCheckpoint(
    #     every_n_epochs=1, dirpath=OUTPUT_DIR, filename="lightning_trained"
    # )
    checkpoint_filename = get_next_checkpoint_filename(CHECKPOINT_DIR, "lightning_trained")
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=checkpoint_filename[:-5],  
        every_n_epochs=1,
        save_top_k=1,  
        monitor="valid_loss",  
        mode="min"
    )
    logger = CSVLogger(OUTPUT_DIR, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator=DEVICE,
        devices=NUM_DEVICES,
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
    )

    segmodel = SegmentationModel(net, loss, LR)
    preprocess_input = smp.encoders.get_preprocessing_fn(
        ENCODER, pretrained=ENCODER_WEIGHTS
    )
    isn_data = ISNDataModule(
        train_df,
        valid_df,
        test_df,
        BATCH_SIZE,
        IMG_SIZE,
        preprocess_input,
        class_rgb_values,
    )

    # checkpoint_file = os.path.join(CHECKPOINT_DIR, "lightning_trained-v1.ckpt")
    # if os.path.isfile(checkpoint_file):
    #     print("Resuming training from previous checkpoint...")
    #     trainer.fit(segmodel, datamodule=isn_data, ckpt_path=checkpoint_file)
    if os.path.isfile(os.path.join(CHECKPOINT_DIR, checkpoint_filename)):
        print("Resuming training from previous checkpoint...")
        trainer.fit(segmodel, datamodule=isn_data, ckpt_path=checkpoint_filename)
    else:
        print("Starting training from scratch...")
        trainer.fit(segmodel, datamodule=isn_data)


def main():
    train()


if __name__ == "__main__":
    main()
