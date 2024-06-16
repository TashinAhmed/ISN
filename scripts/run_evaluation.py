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
Script to execute the main function from src.training.test module.

This script imports and executes the main function from src.training.test module,
which is responsible for testing the segmentation model using the latest checkpoint.
"""

from src.training.test import main

if __name__ == "__main__":
    # Execute the main function from src.training.test module
    main()
