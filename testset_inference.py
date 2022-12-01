
import time
import os, copy, random
import itertools
import io
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler, ConcatDataset
import itertools
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torchmetrics.functional import precision_recall
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.classification import MulticlassAUROC, MulticlassF1Score, MulticlassPrecision

import torchmetrics
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc

import warnings
warnings.filterwarnings(action='ignore')
# warnings.filterwarnings('UndefinedMetricWarning')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')

from options import args_parser
from models import EfficientNet, ResNet, VGG, custom_EN_b0
from skin_cancer_dataset import SkinCancer


from sklearn import metrics
from sklearn.metrics import confusion_matrix
# import tensorflow as tf

import math

    
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest',     cmap='summer')
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

    
# def train_epoch(model,device,dataloader,loss_fn,optimizer):
#     train_loss,train_correct=0.0,0
#     model.train()
#     for images, labels in dataloader:

#         images,labels = images.to(device),labels.to(device)
#         optimizer.zero_grad()
#         output = model(images)
#         loss = loss_fn(output,labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * images.size(0)
#         scores, predictions = torch.max(output.data, 1)
#         train_correct += (predictions == labels).sum().item()

#     return train_loss,train_correct
  
# def valid_epoch(model,device,dataloader,loss_fn):
#     valid_loss, val_correct = 0.0, 0
#     model.eval()
#     y_true,y_pred = [], []
#     for images, labels in dataloader:

#         images,labels = images.to(device),labels.to(device)
#         output = model(images)
#         loss=loss_fn(output,labels)
#         valid_loss+=loss.item()*images.size(0)
#         scores, predictions = torch.max(output.data,1)
        
#         val_correct+=(predictions == labels).sum().item()
        
#         y_true.extend(labels.cpu().numpy())
#         y_pred.extend(predictions.cpu().numpy())
        
        
#     # y_true = np.array(y_true)
#     # y_pred = np.array(y_pred)
#     cf_matrix = confusion_matrix(y_true, y_pred)
    
#     return valid_loss,val_correct, cf_matrix




def get_metrics(y_true, probs):
    metrics = {'f1':[],
               'precision':[],
               'recall':[],
               'auc':[]}
    
    # for i in range(len(y_true)):
    
    f_l = [f1_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
    p_l = [precision_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
    r_l = [recall_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
    
    
        
    metrics['f1'].append(sum(f_l)/len(f_l))
    # p,r = precision_recall(sum(f_l)/len(f_l))
    metrics['precision'].append(sum(p_l)/len(p_l))
    # metrics['recall'].append(r.numpy())
    metrics['recall'].append(sum(r_l)/len(r_l))
    
    df_m = pd.DataFrame(metrics)
    
    return df_m



def get_metrics(y_true, probs):
    # metrics = {'f1':[],
    #            'precision':[],
    #            'recall':[],
    #            'auc':[]}
    y_true = torch.tensor(y_true).to(device)
    
    
    # for i in range(len(y_true)):
        
    f1 = f1_score(probs.detach(), y_true)
    print('f1 in function: ',f1)
    p,r = precision_recall(probs.detach().to(device), y_true, average='weighted', num_classes=9)
    print('p,r: ',p,r)
 
    return f1, p,r




def test_inference(model, testloader):
    
    """ Returns the test accuracy and loss.
    """
    # torch.cuda.empty_cache()
    
    m = nn.Softmax(dim=1)
    
    loss, total, correct = 0.0, 0.0, 0.0
    # len(probs)

    try:
        device = 'mps' 
    except:
        device = 'cpu' 
    
    criterion = nn.CrossEntropyLoss().to(device)
    # testloader = DataLoader(test_dataset, batch_size=8,
    #                         shuffle=False)
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    y_t, y_p = [], []
    # probs = []
    auc=[]
    f1,p,r=[],[],[]
    df = pd.DataFrame(columns=['true', 'prob'])
   
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        

        # Inference
        outputs = model(images)
        # probs= m(outputs)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
            

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        
        
        pred_label = pred_labels.view(-1)
        

        #[0, 1, 2, 3 ..]
        y_true.append(labels.cpu())
    
        #[0, 1, 2, 3 ..]
        y_pred.append(pred_label.cpu())
        
        y_t.extend(labels.cpu().numpy())
        y_p.extend(pred_labels.cpu().numpy())
        
        
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        
    
    # p_s,r_s = precision_recall(y_pred.cpu(), y_true.cpu(), average='weighted', num_classes=9)
    accuracy = correct/total
    auc_s = 0
    return accuracy, loss, y_true, y_pred, y_t, y_p








if __name__ == '__main__':
    
    args = args_parser()
    #exp_details(args)

    try:
        device = 'mps' 
    except:
        device = 'cpu' 
    
    
    k=5
    splits=KFold(n_splits=k,shuffle=True,random_state=42)
    
    
# ======================= DATA ======================= #
    # if not args.federated:
    data_dir = '../../d_5' #    PASTE THE PATH TO THE TEST DATA HERE


    dataset = SkinCancer(os.path.join(data_dir,'train'), transform=None)
    dataset_size = len(dataset)
    
    class_names =  [i.split('/')[-1] for i in dataset.classes]

        
    
    
# ======================= Model | Loss Function | Optimizer ======================= # 

    if args.model == 'efficientnet':
        
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1].out_features = 9
        
    elif args.model == 'resnet':
        model = ResNet(args)
    
    elif args.model == 'vgg19':
        model = VGG(args)
    
    elif args.model == 'custom_EN_b0':
        model = custom_EN_b0(args)


    criterion = nn.CrossEntropyLoss()
    
    batch_size = 6
    

    
    
    
    start_t = time.time()
    fold_his = {}
    class_names = dataset.classes
    
    start_t = time.time()
    
    # copy weights
    # MODEL_WEIGHTS = copy.deepcopy(model.state_dict())
    PATH = 'NAME OF THE SAVED WEIGHTS THE CUSTOM MODEL WAS TRAINED ON HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!.pth'
    best_acc = 0.0
    


    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold))
        
        try:
            device = 'mps' 
        except:
            device = 'cpu' 
        
        # model.load_state_dict(MODEL_WEIGHTS)
        model.load_state_dict(torch.load(PATH))
        model.to(device)

        # train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        # train_dataset = torch.utils.data.Subset(dataset, train_sampler.indices)
        test_dataset = torch.utils.data.Subset(dataset, test_sampler.indices)
        #test_loader = DataLoader(test_set, batch_size=16,shuffle=False)


        # train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        
        # Set optimizer for the local updates
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9)
        elif args.optimizer == 'adamx':
            optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
            

        history = {#'train_loss': [], 'train_acc': [],
                   'test_loss': [], 'test_acc': []}



        local_weights=[]


        
        print(f'Model Initialized for {fold}...')
    # ======================= Train per fold ======================= #
        for epoch in range(args.epochs):
            # train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer)
            test_loss, test_correct, cf_matrix=valid_epoch(model,device,test_loader,criterion)

            # train_loss = train_loss / len(train_loader.sampler)
            # train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100







            print("Epoch:{}/{}\nAVG Training Loss:{:.3f} \t Testing Loss:{:.3f}\nAVG Training Acc: {:.2f} % \t Testing Acc {:.2f} % ".format(epoch, args.epochs, 
                                                                                                                                             #train_loss,  test_loss, 
                                                                                                                                             train_acc,  test_acc))
            print('*'*50)


            # history['train_loss'].append(train_loss)
            # history['train_acc'].append(train_acc)

            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)


            
                 # Test inference after completion of training
        # torch.cuda.empty_cache()
        test_acc, test_loss, y_true, y_pred, y_t, y_p = test_inference(model, test_loader)
        
        f_l = [f1_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
        p_l = [precision_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
        r_l = [recall_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
        # auc_l = [roc_auc_score(p.cpu().numpy(), t.cpu().numpy(), multi_class='ovr') for t,p in zip(y_true,y_pred)]

        f1_avg = sum(f_l)/len(f_l)
        # print('f1:', f1_s)
        p_avg = sum(p_l)/len(p_l)
        r_avg = sum(r_l)/len(r_l)
        
        save_fig = f'../skew/skew_cf/{model._get_name()}c5_fold_{fold}.png'
        cf_fold = plot_confusion_matrix(cf_matrix, class_names=class_names)
        cf_fold.savefig(save_fig)
                
        print('*'*50)
        
        
        with open('../skew/metrics5_per_fold.txt','a+') as f:
                    f.write(f'Fold: {str(fold)}\tF1-Score: {round(f1_avg, 2)}\tPrecision: {str(round(p_avg,2))}\tRecall: {str(round(r_avg,2))}\tAccuracy: {str(round(test_acc,2))}\tLoss: {str(round(test_loss,2))}\n')
                        
        
    # ======================= Save per fold ======================= #
        cf_figure = plot_confusion_matrix(cf_matrix, class_names)
        np.save(f'../skew/skew_cf/{model._get_name()}_fold_{fold}_epoch_{epoch}_5TST.npy', cf_matrix)

        # cf_image = plot_to_image(cf_figure)
        save_fig = f'../skew/skew_cf/{model._get_name()}_fold_{fold}_epoch_{epoch}_5TST.png'
        cf_figure.savefig(save_fig)






        save_df = f'../skew/skew_cv/{model._get_name()}_{args.optimizer}_fold_{fold}_5TST.csv'


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
            torch.save(model.state_dict(), f'../skew/skew_models/{model._get_name()}_{args.optimizer}_5TST.pth')




    # ======================= Save fold history ======================= #

    dff = pd.DataFrame.from_dict({(i,j): fold_his[i][j] 
                               for i in fold_his.keys() 
                               for j in fold_his[i].keys()},
                           orient='columns')

    dff.to_csv(f'../skew/skew_cv/{model._get_name()}_{args.optimizer}_{k}CV_{args.epochs}EPOCHS_5TST.csv')


    end_train = time.time()
    time_elapsed = start_t - end_train


    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
