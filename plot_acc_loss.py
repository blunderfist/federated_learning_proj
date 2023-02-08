import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_metric(data, model_name, metric, save_path):

	plt.plot(data[f'train_{metric}'], label = "Train")
	plt.plot(data[f'test_{metric}'], label = "Test")
	plt.legend()
	plt.title(f"{model_name} {metric}")
	plt.xlabel("Epochs")
	plt.ylabel(metric.title())
	plt.xticks(range(len(data[f'test_{metric}'])))
	# plt.show()
	plt.savefig(os.path.join(save_path, f'{model_name}_{metric}.png'), bbox_inches = 'tight')

def main():
	"""plots loss and accuracy for model"""
	
	# file_name = input("Enter filename: ")
	# file_name = 'Book1.csv'
	dir_glob = glob(os.path.join(r'skewed_results',r'global_results','**', '*/'), recursive = True)
	for dir_ in dir_glob:
		file_glob = glob(os.path.join(dir_, '*LE*.csv')) # only getting acc and loss csvs
		for f in file_glob:
			f_name = os.path.basename(f).split('.')[0]
			save_path = os.path.dirname(f)
			print(f"Processing {f_name}")

			df = pd.read_csv(f, index_col = 0)
			model_name = "_".join(f_name.split('_')[:2]) # get the name of the model + optimizer ignore kfold and other
			metrics = ['loss', 'acc']
			for metric in metrics:
				plot_metric(df, model_name, metric, save_path)

if __name__ == '__main__':
	main()
