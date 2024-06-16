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

import os
import re

def get_next_checkpoint_filename(checkpoint_dir, base_filename):
    """
    Get the next checkpoint filename with an incremented version number.
    
    Args:
    - checkpoint_dir (str): Directory where checkpoint files are stored.
    - base_filename (str): Base filename for the checkpoint files.

    Returns:
    - str: Next checkpoint filename with an incremented version number.
    """
    version_pattern = re.compile(rf"{base_filename}-v(\d+)\.ckpt")
    max_version = 0

    for filename in os.listdir(checkpoint_dir):
        match = version_pattern.match(filename)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)

    next_version = max_version + 1
    return f"{base_filename}-v{next_version}.ckpt"
