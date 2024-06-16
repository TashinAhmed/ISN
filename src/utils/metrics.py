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
Utility functions for computing segmentation metrics using segmentation_models_pytorch.

This script contains a utility function to compute segmentation metrics:
1. IOU (Intersection over Union) score
2. F1 score
3. Accuracy
4. Recall

The metrics are computed using segmentation_models_pytorch library functions.

Imports:
- segmentation_models_pytorch as smp: Importing the library for segmentation metrics.

Functions:
- compute_metrics(preds, labels): Computes segmentation metrics given predicted and ground truth labels.
- main(): Entry point of the script, prints a message indicating that metrics utilities are ready.

Usage:
- Run this script to use the compute_metrics function for evaluating segmentation model performance.

Exceptions:
- Assumes input predictions and labels are correctly formatted numpy arrays.

"""

import segmentation_models_pytorch as smp


def compute_metrics(preds, labels):
    """
    Compute segmentation metrics (IOU score, F1 score, accuracy, recall).

    Args:
    - preds (np.ndarray): Predicted segmentation mask of shape (B, H, W).
    - labels (np.ndarray): Ground truth segmentation mask of shape (B, H, W).

    Returns:
    - dict: Dictionary containing computed metrics:
    - "iou_score" (float): IOU score.
    - "f1_score" (float): F1 score.
    - "accuracy" (float): Accuracy.
    - "recall" (float): Recall.
    """
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, labels.long(), mode="binary", threshold=0.5
    )
    metrics = {
        "iou_score": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"),
        "f1_score": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro"),
        "accuracy": smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro"),
        "recall": smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise"),
    }
    return metrics


def main():
    print("Metrics utilities are ready to use.")


if __name__ == "__main__":
    main()
