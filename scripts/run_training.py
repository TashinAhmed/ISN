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
Script to execute the main function from src.training.train module.

This script imports and executes the main function from src.training.train module,
which is responsible for training the segmentation model and saving checkpoints.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.train import main

if __name__ == "__main__":
    # Execute the main function from src.training.train module
    main()
