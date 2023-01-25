# federated_learning_proj

---

Code for federated learning project.

Contents:
 - [About](##-About)
 - [Dataset](##-Dataset)
 - [To run](##-To-Run)
 
 ## About
 
 This is some code I've written or edited from another project on federated learning. Myself and the original author are working on a subset of the author's original problem as a small research project.

 Some additional helper files have been included.
 
 ## Dataset
 
 Dataset available from Kaggle https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
 [Go to dataset here](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
 
 For this project the data must be skewed across each client. One client will have 50% of a class, chosen at random, and the remaineder of that class is split evenly across the remaining clients. split_and_skew_the_data.py will perform this for you but first the data must be downloaded and extracted as follows.
 
 It's a zipped file and the code to skew it will look for data in the /data directory, so make sure it's unzipped into this directory:
- Linux ```unzip archive.zip -d data```
- Powershell ```Expand-Archive -Path archive.zip -DestinationPath data```
- MAC OS ```ditto -xk archive.zip data```
 
 To skew the dataset run this or something similar.
 - ```py split_and_skew_the_data.py```

This script must be in the same directory containing /data.
 
 ## To run
 
 From the terminal enter <code>py .\s2kew_fl_kfold2.py --model custom_EN_b0_v3</code>
 
