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

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import ISNSet
from src.config import WORKERS


class ISNDataModule(pl.LightningDataModule):
    """
    LightningDataModule for ISN dataset.

    This module handles loading and preprocessing of ISN dataset for training,
    validation, and testing using PyTorch Lightning DataLoader.
     
    Initializes ISNDataModule with dataset information and parameters.

    Args:
    - train_df (pd.DataFrame): DataFrame containing training set metadata.
    - valid_df (pd.DataFrame): DataFrame containing validation set metadata.
    - test_df (pd.DataFrame): DataFrame containing test set metadata.
    - batch_size (int): Batch size for DataLoader (default: 2).
    - img_size (tuple): Tuple specifying image size (default: (256, 256)).
    - preprocess_fn (callable): Preprocessing function for input images (default: None).
    - class_rgb_values (list): List of RGB values for dataset classes (default: None).
        
    """
    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        batch_size=2,
        img_size=(256, 256),
        preprocess_fn=None,
        class_rgb_values=None,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=self.img_size),
            ]
        )
        self.preprocess_fn = preprocess_fn
        self.class_rgb_values = class_rgb_values

    def setup(self, stage=None):
        """
        Setup datasets for training, validation, and testing.

        Args:
        - stage (str): Stage of setup ('fit' for training, 'test' for testing).
        """
        self.trainset = ISNSet(
            self.train_df,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
            class_rgb_values=self.class_rgb_values,
        )
        self.validset = ISNSet(
            self.valid_df,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
            class_rgb_values=self.class_rgb_values,
        )
        self.testset = ISNSet(
            self.test_df,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
            class_rgb_values=self.class_rgb_values,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=True,
        )


def main():
    print("Data module is ready to use.")


if __name__ == "__main__":
    main()
