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
PyTorch Lightning Segmentation Model for training, validation, and testing.

This module defines a segmentation model using segmentation_models_pytorch, incorporating metrics
such as IoU score, F1 score, accuracy, and recall for evaluation during training, validation, and testing.
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import torch
from torch import nn


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
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, labels.long(), mode="binary", threshold=0.5
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        return {
            "iou_score": iou_score,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "recall": recall,
        }

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
        - batch (tuple): Batch of input images and labels.
        - batch_idx (int): Index of the batch.

        Returns:
        - torch.Tensor: Loss tensor for the batch.
        """
        imgs, labels = batch
        preds = self(imgs)
        metrics = self.shared_step(preds, labels)
        self.log("train_iou_score", metrics["iou_score"], on_epoch=True)
        self.log("train_f1_score", metrics["f1_score"], on_epoch=True)
        self.log("train_accuracy", metrics["accuracy"], on_epoch=True)
        self.log("train_recall", metrics["recall"], on_epoch=True)
        loss = self.loss(preds, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
        - batch (tuple): Batch of input images and labels.
        - batch_idx (int): Index of the batch.
        """
        imgs, labels = batch
        preds = self(imgs)
        metrics = self.shared_step(preds, labels)
        self.log("valid_iou_score", metrics["iou_score"], on_epoch=True)
        self.log("valid_f1_score", metrics["f1_score"], on_epoch=True)
        self.log("valid_accuracy", metrics["accuracy"], on_epoch=True)
        self.log("valid_recall", metrics["recall"], on_epoch=True)
        loss = self.loss(preds, labels)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        """
        Defines the test step.

        Args:
        - batch (tuple): Batch of input images and labels.
        - batch_idx (int): Index of the batch.
        """
        imgs, labels = batch
        preds = self(imgs)
        metrics = self.shared_step(preds, labels)
        self.log("test_iou_score", metrics["iou_score"], on_epoch=True)
        self.log("test_f1_score", metrics["f1_score"], on_epoch=True)
        self.log("test_accuracy", metrics["accuracy"], on_epoch=True)
        self.log("test_recall", metrics["recall"], on_epoch=True)
        loss = self.loss(preds, labels)
        self.log("test_loss", loss)

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
