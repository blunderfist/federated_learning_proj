#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import os, glob, shutil
from distutils.dir_util import copy_tree

def combine_train_test(path_to_data = None):
	"""Combines training and test sets to create larger set for CV
	params:
		path_to_data: path to directory containing data
	returns:
		None: creates combined directory containing merged training and testing directories
	"""

	if path_to_data is None:
		data_path = os.path.join(os.path.dirname(os.getcwd()), 'skin_cancer_data_fed')

	folders = glob.glob(data_path + '/*')
	print("Getting folders")
	os.mkdir(os.path.join(data_path, 'combined'))
	combined = os.path.join(data_path, 'combined')
	print("Created combined directory")
	for folder in folders:

		print(f"Working on {os.path.basename(folder)} data...")
		if os.path.basename(folder) == 'test':
			from_directory = folder
			to_directory = combined
			copy_tree(from_directory, to_directory)
		print(f"{os.path.basename(folder)} data copied\n")
		
		if os.path.basename(folder) == 'train':
			test_folders_glob = glob.glob(folder + '/*')
			for folder in test_folders_glob:
				class_label = os.path.basename(folder)
				combined_dir = os.path.join(os.path.dirname(os.getcwd()), 'skin_cancer_data_fed', 'combined', class_label)
				if not os.path.isdir(combined_dir):
					print(f"There's an error working with {class_label}")
					return False
				current_dir = os.path.join(os.path.dirname(os.getcwd()), 'skin_cancer_data_fed', 'train', class_label)
				current_dir_contents = os.listdir(current_dir)
				print(f"Copying {len(current_dir_contents)} files from {class_label}")
				for fname in current_dir_contents:
					shutil.copy2(os.path.join(current_dir, fname), combined_dir)

	print("Train and test folders merged into '/combined'\n")

combine_train_test()