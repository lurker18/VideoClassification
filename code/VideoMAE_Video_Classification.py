# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:24:01 2024

@author: Nova18
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# Define the directory where your folders with video files are located
video_directory = 'D:/HuggingFace/datasets/UCF101/all_data'

# Define the directories where you want to save your train/validation/test sets
train_dir = 'D:/HuggingFace/datasets/UCF101/train'
val_dir = 'D:/HuggingFace/datasets/UCF101/validation'
test_dir = 'D:/HuggingFace/datasets/UCF101/test'

# Create these directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all folders in the directory
folders = [f for f in os.listdir(video_directory) if os.path.isdir(os.path.join(video_directory, f))]

for folder in folders:
    # List all files in each folder
    files = os.listdir(os.path.join(video_directory, folder))
    
    # Split files into training (60%), validation (20%), and test (20%) sets
    train_files, test_val_files = train_test_split(files, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)
    
    # Function to copy files to the designated directory
    def copy_files(files_list, set_directory):
        set_folder = os.path.join(set_directory, folder)
        os.makedirs(set_folder, exist_ok=True)
        for file_name in files_list:
            shutil.copy2(os.path.join(video_directory, folder, file_name), set_folder)
    
    # Copy files to their respective set directories
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)


