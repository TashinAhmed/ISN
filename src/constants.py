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

DATA_DIR = "./data/preprocessed"
CLASS_DICT_FILE = os.path.join(DATA_DIR, "class_dict.csv")


def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Class dictionary file: {CLASS_DICT_FILE}")


if __name__ == "__main__":
    main()
