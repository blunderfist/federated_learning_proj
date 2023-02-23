import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def main():
	files = glob(os.path.join("skewed_results", "models", "*/","inference",'*.csv'))
	for file in files:
	    print(f"opening {file}")
	    model_name = file.split('\\')[2].replace("_results","")
	    save_path = os.path.dirname(file)
	    df = pd.read_csv(file)
	    y_true = df['y_true']
	    y_pred = df['y_pred']
	    cm = confusion_matrix(y_true, y_pred)
	    plt.figure(figsize = (10,7))
	    plt.title(f'Confusion Matrix\n{model_name} inference')
	    sns.heatmap(cm, annot=True)
	    plt.savefig(os.path.join(save_path, f'{model_name}_inference.png'))
	    print(f"Saved image to {os.path.join(save_path, f'{model_name}_inference.png')}")

if __name__=="__main__":
	main()