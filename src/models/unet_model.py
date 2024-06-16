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
Utilized UNet++ model using segmentation_models_pytorch library.

This script defines a function to create a UNet++ model for semantic segmentation.
It utilizes configurations imported from src.config module.

Attributes:
- ENCODER (str): Name of the encoder backbone.
- ENCODER_WEIGHTS (str): Type of weights to use for encoder initialization.
- CHANNELS (int): Number of input channels.
- CLASSES (int): Number of output classes.
- ACTIVATION (str): Activation function to use for the final layer.

Functions:
- create_unet_model(): Creates and returns a UNet++ model using the specified configurations.
- main(): Entry point of the script that demonstrates the creation of the UNet model.
"""

import segmentation_models_pytorch as smp
from src.config import ENCODER, ENCODER_WEIGHTS, CHANNELS, CLASSES, ACTIVATION


def create_unet_model():
    """
    Create UNet++ model based on the imported configurations.

    Returns:
    - torch.nn.Module: UNet++ model instance.
    """
    net = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=CHANNELS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
    return net


def main():
    model = create_unet_model()
    print("UNet model created successfully.")


if __name__ == "__main__":
    main()
