# federated_learning_proj

---

Code for federated learning project.

Contents:
 - About
 - To run
 
 ## About
 
 This is code I've written or edited from another project on federated learning. Myself and the original author are working on a subset of the author's original problem as a small research project.
 
 The original work uses federated learning to train a model on skin cancer image data from 5 clients.
 
 In this version we are attempting to build a model identified in literature to improve skin cancer image data classification. The final model will be compared against the other models used in the federated learning project and is based off of EfficientNet b0. The last 3 layers of EfficientNet b0 are removed and have been replaced with the improvements.
 
 Some additional helper files have been included.
 
 ## To run
 
 From the terminal enter <code>py .\s2kew_fl_kfold2.py --model custom_EN_b0</code>
 
