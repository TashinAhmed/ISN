#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By   : Tashin Ahmed
# Created Date : "17/06/2024"
# email        : tashinahmed.contact@gmail.com
# copyright    : MIT License Copyright (c) 2024 Tashin Ahmed
# version      : "0.0.1"
# status       : "PoC"
# ----------------------------------------------------------------------------

"""
PyTorch Lightning Segmentation Model for training, validation, and testing.

This module defines a segmentation model using segmentation_models_pytorch, incorporating metrics
such as IoU score, F1 score, accuracy, and recall for evaluation during training, validation, and testing.
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import torch
from torch import nn

from src.utils.metrics import compute_metrics, log_metrics_and_loss


class SegmentationModel(pl.LightningModule):
    """
    LightningModule for segmentation model training, validation, and testing.

    Args:
    - net (torch.nn.Module): Segmentation network model.
    - loss (torch.nn.Module): Loss function for training.
    - lr (float): Learning rate for optimizer.

    Methods:
    - forward(x): Forward pass through the network.
    - shared_step(preds, labels): Calculates segmentation metrics (IoU, F1, accuracy, recall).
    - training_step(batch, batch_idx): Defines the training step.
    - validation_step(batch, batch_idx): Defines the validation step.
    - test_step(batch, batch_idx): Defines the test step.
    - configure_optimizers(): Configures the optimizer for training.
    """

    def __init__(self, net, loss, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss = loss
        self.net = net

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor from the network.
        """
        return self.net(x)

    def shared_step(self, preds, labels):
        """
        Calculates segmentation metrics (IoU, F1, accuracy, recall).

        Args:
        - preds (torch.Tensor): Predictions from the network.
        - labels (torch.Tensor): Ground truth labels.

        Returns:
        - dict: Dictionary containing computed metrics.
        """

        metrics_dict = compute_metrics(preds, labels)
        return metrics_dict

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
        - batch (tuple): Batch of input images and labels.
        - batch_idx (int): Index of the batch.

        Returns:
        - torch.Tensor: Loss tensor for the batch.
        """

        return log_metrics_and_loss(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
        - batch (tuple): Batch of input images and labels.
        - batch_idx (int): Index of the batch.
        """
        
        log_metrics_and_loss(self, batch, "valid")

    def test_step(self, batch, batch_idx):
        """
        Defines the test step.

        Args:
        - batch (tuple): Batch of input images and labels.
        - batch_idx (int): Index of the batch.
        """

        log_metrics_and_loss(self, batch, "test")

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
        - list: List of optimizers.
        """
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return [opt]


def main():
    print("Segmentation model is ready to use.")


if __name__ == "__main__":
    main()
