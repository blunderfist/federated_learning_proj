import time
import os, copy, random
import itertools
import io
# import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler, ConcatDataset

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score, MulticlassPrecision

import torchmetrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split



from options import args_parser
from models import EfficientNet, ResNet, VGG
from skin_cancer_dataset import SkinCancer


from sklearn import metrics
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import math
# wandb.init(project="VGG19")

# wandb.init(
#     # Set the project where this run will be logged
#     project="Baseline-VGG", 
#     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#     # Track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.001,
#     "architecture": "VGG_19",
#     "dataset": "Skin Cancer",
#     "epochs": 15,
#     })
    

    
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45,fontsize=8,horizontalalignment='right')
    plt.yticks(tick_marks, class_names,fontsize=8)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color,fontsize=7)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

    
def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct
  
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    y_true,y_pred = [], []
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        
        val_correct+=(predictions == labels).sum().item()
        

        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        
        
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred)
    



    return valid_loss,val_correct, cf_matrix

def test_inference(model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'mps' 
    criterion = nn.CrossEntropyLoss().to(device)
    
    f_score=[]
    y_true,y_pred = [], []

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        # pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        
        sf = torch.nn.Softmax(dim=0)
        prob = sf(outputs)
        
        
        yt = labels.cpu().numpy()
        yp = prob.detach().cpu().numpy()
        
#         print(f'YT  {type(yt)} YP: {type(yp)}')
#         score = roc_auc_score(yt, yp, average='weighted')
        
#         print(f"AUC Score : >>>>>>>>>>>>>   {score} <<<<<<<<<<<")
        

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
# 
    accuracy = correct/total
    return accuracy, loss, y_true, y_pred

if __name__ == '__main__':
    args = args_parser()
    
    device = 'mps'
    
    
    k=2
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    
    
# ======================= DATA ======================= #
    # if not args.federated:
    data_dir = '../../skin_cancer_data_fed'


    dataset = SkinCancer(os.path.join(data_dir,'train'), transform=None)



    dataset_size = len(dataset)

        
    
    
# ======================= Model | Loss Function | Optimizer ======================= # 

    if args.model == 'efficientnet':
        
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1].out_features = 9
        
    elif args.model == 'resnet':
        model = ResNet(args)
    
    elif args.model == 'vgg19':
        model = VGG(args)
        


    criterion = nn.CrossEntropyLoss()
    
    batch_size = 16
    

    
    
    
    start_t = time.time()
    fold_his = {}
    class_names = dataset.classes
    
    start_t = time.time()
    
    # copy weights
    MODEL_WEIGHTS = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold))
        
        device = 'mps'
        
        model.load_state_dict(MODEL_WEIGHTS)
        model.to(device)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


        
        
        # Set optimizer for the local updates
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9)
        elif args.optimizer == 'adamx':
            optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
            

        history = {'train_loss': [], 'train_acc': [],
                   'test_loss': [], 'test_acc': []}



        


    # ======================= Train per fold ======================= #
        for epoch in range(args.epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss, test_correct, cf_matrix=valid_epoch(model,device,test_loader,criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100







            print("Epoch:{}/{}\nAVG Training Loss:{:.3f} \t Testing Loss:{:.3f}\nAVG Training Acc: {:.2f} % \t Testing Acc {:.2f} % ".format(epoch, args.epochs, 
                                                                                                                                             train_loss,  test_loss, 
                                                                                                                                             train_acc,  test_acc))




            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)



    # ======================= Save per fold ======================= #
        cf_figure = plot_confusion_matrix(cf_matrix, class_names)
        np.save(f'../save_new_baseline/cf_matrix/{model._get_name()}_fold_{fold}_epoch_{epoch}_TST.npy', cf_matrix)

        # cf_image = plot_to_image(cf_figure)
        save_fig = f'../save_new_baseline/cf_matrix/{model._get_name()}_fold_{fold}_epoch_{epoch}_TST.png'
        cf_figure.savefig(save_fig)






        save_df = f'../save_new_baseline/baseline_crossvalidation/{model._get_name()}_{args.optimizer}_fold_{fold}_TST.csv'


        df_fold = pd.DataFrame(history)
        df_fold.to_csv(save_df)
        # print(df_fold)
        fold_his['fold{}'.format(fold+1)] = history


    
    # ======================= Save model if new high accuracy ======================= #
        if test_acc > best_acc:
            print('#'*25)
            print('New High Acc: ', test_acc)
            print('#'*25)
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'../save_new_baseline/models/{model._get_name()}_{args.optimizer}_TST.pth')




    # ======================= Save fold history ======================= #

    dff = pd.DataFrame.from_dict({(i,j): fold_his[i][j] 
                               for i in fold_his.keys() 
                               for j in fold_his[i].keys()},
                           orient='columns')

    dff.to_csv(f'../save_new_baseline/baseline_crossvalidation/{model._get_name()}_{args.optimizer}_{k}CV_{args.epochs}EPOCHS_TST.csv')


    end_train = time.time()
    time_elapsed = start_t - end_train


    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


    
    