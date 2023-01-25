# federated_learning_proj

---

Code for federated learning project.

Contents:
 - About
 - To run
 
 ## About
 
 This is some code I've written or edited from another project on federated learning. Myself and the original author are working on a subset of the author's original problem as a small research project.

 Some additional helper files have been included.
 
 ## Dataset
 
 Dataset available from Kaggle https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
 [Go to dataset here](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
 
 It's a zipped file and the code to skew it will look for data in the /data directory, so make sure it's unzipped into this directory:
- Linux ```unzip archive.zip -d data```
- Powershell ```Expand-Archive -Path archive.zip -DestinationPath data```
- MAC OS ```ditto -xk archive.zip data```
 

 
 ## To run
 
 From the terminal enter <code>py .\s2kew_fl_kfold2.py --model custom_EN_b0_v3</code>
 
