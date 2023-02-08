import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_metric(data, model_name, metric, date_time):

	plt.plot(data[f'train_{metric}'], label = "Train")
	plt.plot(data[f'test_{metric}'], label = "Test")
	plt.legend()
	plt.title(f"{model_name} {metric}")
	plt.xlabel("Epochs")
	plt.ylabel(metric.title())
	plt.xticks(range(len(data[f'test_{metric}'])))
	# plt.show()
	plt.savefig(f'{date_time}_{model_name}_{metric}.png', bbox_inches = 'tight')

def main():
	"""plots loss and accuracy for model"""
	
	file_name = input("Enter filename: ")
	# file_name = 'Book1.csv'
	df = pd.read_csv(file_name, index_col = 0)
	model_name = file_name.split('_')[0] # get the name of the model
	date_time = "".join(str(datetime.now()).split('.')[0].split(':')) # getting current date and time for reference to when last run
	metrics = ['loss', 'acc']
	for metric in metrics:
		plot_metric(df, model_name, metric, date_time)

if __name__ == '__main__':
	main()
