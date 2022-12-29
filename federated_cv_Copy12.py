import os,sys
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from skin_cancer_dataset import SkinCancer
import matplotlib.pyplot as plt
import itertools
import io
import torchmetrics
# from torchmetrics.functional import precision_recall
import torch
from torch import nn
# from tensorboardX import SummaryWriter
import torchvision
from options import args_parser
from _update import LocalUpdate
# from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from torch.utils.data import Dataset, DataLoader
import datetime

from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
from models import EfficientNet, ResNet, VGG, custom_EN_b0, custom_EN_b0_v2, custom_EN_b0_v3
# from pycm import *
import pycm

import warnings
warnings.filterwarnings(action = 'ignore')
# warnings.filterwarnings('UndefinedMetricWarning')

def skin_cancer_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    arr = np.array([])
    dict_users = {i: [] for i in range(5)}
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        ch = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i].append(ch)
        dict_users[i] = np.asarray(ch)
        # all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# def skin_cancer_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     arr = np.array([])
#     dict_users = {i: [] for i in range(5)}
#     all_idxs = [i for i in range(len(dataset))]
#     for i in range(num_users):
#         ch = np.random.choice(all_idxs, num_items, replace=False)
#         dict_users[i].append(ch)
#         dict_users[i] = np.asarray(ch)
#         # all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def test_inference(model, testloader):
    
    """ Returns the test accuracy and loss.
    """
    # torch.cuda.empty_cache()
    
    m = nn.Softmax(dim=1)
    
    loss, total, correct = 0.0, 0.0, 0.0
    # len(probs)
    # try:
    #     if torch.cuda.is_available():
    #         device = torch.device("cuda:0" else "cpu")
    #     # device = 'mps' 
    # except:
    # device = 'mps'
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

    # return accuracy, loss, f1_s, p_s, r_s, auc_s



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
    
    # f1_avg = df_m['f1'].mean()
    # p_avg = df_m['precision'].mean()
    # r_avg = df_m['recall'].mean()
    # auc_avg = df_m['auc'].mean()
    
    # print(f1_avg, p_avg, r_avg, auc_avg)
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
    # metrics['precision'].append(p.numpy())
    # metrics['recall'].append(r.numpy())
    # auc = auroc(probs.detach().to(device), y_true[-1])    
#     df_m = pd.DataFrame(metrics)
    
#     f1_avg = df_m['f1'].mean()
#     p_avg = df_m['precision'].mean()
#     r_avg = df_m['recall'].mean()
#     auc_avg = df_m['auc'].mean()
    
    # print(f1_avg, p_avg, r_avg, auc_avg)
    return f1, p,r





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
        color = "white" if cm[i, j] < threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color,fontsize=7)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

    # return cf_matrix
    




    ######################################################################################################################################

def build_train_set_lst():
    """builds list of all skewed datasets"""
    # skewed_datasets = ['d_1','d_2','d_3','d_4','d_5']
    skewed_datasets = [f'client_{x}' for x in range(1, 6)]
    train_set_lst = []
    for i in range(len(skewed_datasets)):
        # use this first one on DANICA's computer
        # tmp_train_set =  SkinCancer(os.path.join(f'../../{skewed_datasets[i]}','train'), transform = None)
        # work comp
        # tmp_train_set =  SkinCancer(os.path.join(f'{skewed_datasets[i]}','train'), transform = None)
        # my laptop
        # tmp_train_set =  SkinCancer(os.path.join('..',f'{skewed_datasets[i]}','train'), transform = None)
        # hpc
        tmp_train_set =  SkinCancer(os.path.join('data_skewed',f'{skewed_datasets[i]}','train'), transform = None)
        train_set_lst.append(tmp_train_set)
    # print("train_set_lst",train_set_lst)
    # skewed_lst = {k:v for k,v in zip(train_set_lst, range(len(skewed_datasets)))} # works but modifying below
    # skewed_lst = {k:v for k,v in zip(skewed_datasets, train_set_lst)}
    skewed_lst = {k:v for k,v in zip(range(len(skewed_datasets)), train_set_lst)}
    # print("skewed_lst",skewed_lst) # this will tell us what dataset is from which folder for sanity check
    return train_set_lst, skewed_lst
##################################


def skin_cancer_skewed(dataset, num_users):
    """
    Sample skewed client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # randomly shuffle skewed datasets
    # print("datasets",dataset)
    # np.random.shuffle(dataset)
    # assign randomly shuffled datasets to users in dictionary
    # num users must = number of datasets
    dict_users = {k:v for k,v in zip(range(num_users), dataset)}

    for i in range(num_users):
        tmp = []
        user_dataset =  dict_users[i]
        # print(i, user_dataset)
        all_idxs = [i for i in range(len(user_dataset))]
        # print("all_idxs",np.asarray(all_idxs))
        dict_users[i] = np.asarray(all_idxs)

    return dict_users


def get_train_set(client_idx):
    """gets skincancer object for each client with path to their dataset"""
    # skewed_datasets = ['d_1','d_2','d_3','d_4','d_5']
    skewed_datasets = [f'client_{x}' for x in range(1, 6)]
    # danicas comp
    # train_set = SkinCancer(os.path.join(f'../../{skewed_datasets[client_idx]}','train'), transform = None)
    # my comp
    # train_set = SkinCancer(os.path.join('..',f'{skewed_datasets[client_idx]}','train'), transform = None)
    # hpc
    train_set = SkinCancer(os.path.join('data_skewed',f'{skewed_datasets[client_idx]}','train'), transform = None)

    return train_set



###########################################################################################
# KFOLD CODE
###########################################################################################

def splice(lst, start_test, stop_test):
    """
    spliced input data for kfold cv
    params:
        lst: list of indices for data
        start_test: index to start split at
        stop_test: index to end split at
    returns:
        train: indices for training set, the test set is configured elsewhere
    """
    lst = list(lst)
    start_train = lst[:start_test]
    stop_train = lst[stop_test:]
    start_train.extend(stop_train)
    train = np.asarray(start_train, dtype = 'int32') # run into problems later if not array
    return train

###########################################################################################
# END KFOLD CODE
###########################################################################################

def sanity_check_datasets(clients, skewed_ds):
    """maps clients to dataset folders for santity checking all datasets are being used"""

    # print(f"clients {clients}") # prints entire client folder indexes
    print(f"clients {clients.keys()}") # prints just the keys
    print(f"skewed_ds {skewed_ds}")



#########################################################################################################################


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    # path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # try:
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     # device = 'mps' if args.gpu else 'cpu'
    # except:
    #     device = 'cpu'
    # device = 'mps'
    device = 'cpu'
    # load dataset and user groups
    # train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # train_set =  SkinCancer(os.path.join('../../skin_cancer_data_fed','train'), transform=None)
    # train_set =  SkinCancer(os.path.join(f'../../{d_?}','train'), transform=None)
#     ######################################################################################################################################

# ##################################
# changing the train set to be a list of training sets each from a skewed dataset
# these will be accessed by indexing in the kfold below
    train_set, skewed_datasets = build_train_set_lst()
    # print(f"skewed_datasets{skewed_datasets}")
    # print("\n\n\n\n\n\n\n\nprinting training set so i know what it looks like")
    # for i in range(len(train_set)):
    #     print(train_set[i])
    #     print("len traingset",len(train_set),"\n\n\n\n\n\n\n\n\n")
##################################
    # tmp = train_set[0]
    # print("TRAIN SET",tmp.classes)



    print("train set 0",train_set)
    class_names =  [os.path.basename(i) for i in train_set[0].classes]
    print(f"CLASS NAMES \n\n{class_names}\n\n")
    # test_set = SkinCancer(os.path.join('../../skin_cancer_data_fed','test'), transform=None)





    ######################################################################################################################################
    
    # user_groups = skin_cancer_iid(train_set, 5) # was 2
    user_groups = skin_cancer_skewed(train_set, 2)
    # print("user_groups type",type(user_groups))
    # print("user_groups",user_groups)
    # this is showing us which client has which dataset
    # not easy to tell which is which but we can determine each is accounted for
    # print(f"\n\nSkewed Dataset Ordering\n\n{[v for v in skewed_datasets.values()]}\n\n\n")
    ######################################################################################################################################

#######################################################################

    # sanity_check_datasets(user_groups, skewed_datasets)
#######################################################################

    
    # this was the hardcoded model from before
    # GLOBAL_MODEL = torchvision.models.efficientnet_b0(pretrained=True)
    # GLOBAL_MODEL = custom_EN_b0_v2(pretrained=True)
    # GLOBAL_MODEL = args.model

    # need this section to pull models from models.py
    # this was previously hardcoded to efficientnet 
    if args.model == 'efficientnet':
        
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1].out_features = 9
        
    elif args.model == 'resnet':
        model = ResNet(args)
    
    elif args.model == 'vgg19':
        model = VGG(args)

    elif args.model == 'custom_EN_b0':
        model = custom_EN_b0(args)

    elif args.model == 'custom_EN_b0_v2':
        model = custom_EN_b0_v2(args)

    elif args.model == 'custom_EN_b0_v3':
        model = custom_EN_b0_v3(args)


###################################################################################################################
### FREEZING WEIGHTS HERE
### CHECK WHERE THEY ARE SAVE FOR FINE TUNING NEXT
###################################################################################################################
    c = 0
    for name, param in model.named_parameters():
        if c < 208: # this is where the layers changed, all new layers should be set to True
            param.requires_grad = False
        # print(name, ':', param.requires_grad) # if you want to print and check they are froze uncomment these
        # print("PARAM # ", c)
        c += 1
###################################################################################################################








    GLOBAL_MODEL = model

    # this is changing the output if its not a custom model
    # our models have correct output size, pretrained do not
    custom = ['custom_EN_b0_v2', 'custom_EN_b0_v3']
    if args.model not in custom:
        old_fc = GLOBAL_MODEL.classifier.__getitem__(-1)
        new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 9, bias=True)
        GLOBAL_MODEL.classifier.__setitem__(-1 , new_fc)
    print(type(GLOBAL_MODEL))
    # GLOBAL_MODEL.classifier[1].out_features = 9
    # print(GLOBAL_MODEL.classifier)
    
    

    # Set the model to train and send it to device.
    GLOBAL_MODEL.to(device)
    GLOBAL_MODEL.train()
    # print(global_model)
    
    

    # copy weights
    GLOBAL_MODEL_WEIGHTS = copy.deepcopy(GLOBAL_MODEL.state_dict())
    
    
    # global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    
    test_loss, test_accuracy = [], []
    
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    
    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': []}
    
    history_c={'client_acc':[], 'client_loss':[]}
        
        
    global_model = GLOBAL_MODEL

###############################################################################################################################    
    # k=2
    # k = 5
    # splits = KFold(n_splits = k, shuffle = True, random_state = 42)
    
    best_acc = 0.0
    AVG_WEIGHTS=[]
    


######################MOVIGN THIS TO A LOWER PORTION
    # for fold in range(k):##############################################################

        # This is new idea
        # now the kfold is just running through a range 0-k and using that as the fold #
        # the data is not being split here because it won't work with current setup
        # after a user_idx is selected below, data is split using a 4/5 1/5 split for train, val
        # currently missing function for this but will create


    local_weights=[]
    user_sets=[]
        
    global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
        
    # print(f'Model Initialized for fold {fold}...') # moved down into for loop just below

    k = 2 # K folds
# ######################################################
# ####    BEGIN KFOLD         
    for fold in range(k):
        print(f'Model Initialized for fold {fold}...')
        
# ######################################################
        
        
        for epoch in tqdm(range(int(args.epochs))): # GLOBAL EPOCHS CURRENTLY AT 3
            # print(f'Model Initialized for fold {epoch}...')
            local_weights, local_losses, local_acc = [], [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')



            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            # pick a random client from the range of clients
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            
        # going thru each user in the list without replacement
#         for idx in idxs_users:
#             # print(f'User Index__________{idx} __________')
                

#             # a user has been chosen (idx)
#             # this idx is being used to select the index of the dataset in the list of skewed datasets we created
#             # then we split the dataset we selected

#             # print(f"TRAIN SET IDX\n\n\n\n\n\n{user_groups[idx]}\n\n\n\n\n\n\n")
#             client_data = user_groups[idx][:] # trying to copy all indexes from dataset for given client
#             # print("user_groups[idx]\n\n\n\n\n",type(user_groups[idx]),'\n\n\n\n\n\n\n')
#             np.random.shuffle(client_data)
            
# ######################################################
# ####    BEGIN KFOLD         
#         for fold in range(k):
#             print(f'Model Initialized for fold {epoch}...')
        
# ######################################################
            # for idx in idxs_users:
            for idx in range(args.num_users):

            # print(f'User Index__________{idx} __________')

                client_data = user_groups[idx][:] # trying to copy all indexes from dataset for given client
                print(f'\nuser_groups__{idx}\t\t',type(user_groups[idx]),'\n\n')
                # np.random.shuffle(client_data)

                # this is the new kfold section
                # splicing client data for kfold
                fold_size = int(len(client_data) / k)
                start_test = fold * fold_size # index for beginning of holdout set
                stop_test = start_test + fold_size # index for end of holdout set
                train_idx = splice(client_data, start_test, stop_test) # return list of training set
                val_idx = client_data[start_test:stop_test] # test set


#                 print(f"get_train_set(client_idx): {get_train_set(idx)}")
#                 # print(f"DATA: {client_data}")
#                 print(f"DATA type: {type(client_data)}")
#                 # print(f"DATA train_idx: {train_idx}")
#                 # print(f"DATA val_idx: {val_idx}")
                print(f"\n\nDATA LEN: {len(client_data)}\n\n")
#                 print(f"FOLD SIZE: {fold_size}")
#                 print(f"start_test: {start_test}")
#                 print(f"stop_test: {stop_test}")
#                 print(f"TYPE train_idx: {type(train_idx)}")
#                 print(f"TYPE val_idx: {type(val_idx)}")
#                 prints all of the indexes in the training dataset
#                 different lengths for each set
#                 print(f"TRAIN_IDX\n\n\n\n\n\n{train_idx}\n\n\n\n\n\n\n")

                train_set = get_train_set(idx)
                train_sampler = SubsetRandomSampler(train_idx)
                test_sampler = SubsetRandomSampler(val_idx)

######################################################################################################################
                # added [data_index] here too
                train_dataset = torch.utils.data.Subset(train_set, train_sampler.indices)
                test_set = torch.utils.data.Subset(train_set, test_sampler.indices)
                test_loader = DataLoader(test_set, batch_size=16,shuffle=False)

                


                # print("SKEWED DATASET[IDX]",skewed_datasets[idx])


                # print("client_data",[type(x) for x in client_data if not type(x) == type(np.int32) ])

                # dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
                dataloader = DataLoader(train_set, batch_size=16, sampler=train_sampler)
                print(f'User Index__________{idx} __________')
                if epoch == 0:

                    global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=client_data, weights = GLOBAL_MODEL_WEIGHTS,
                                          model = global_model, logger=None)                    

                
                else:

                    global_model.load_state_dict(global_weights)
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=client_data , weights = GLOBAL_MODEL_WEIGHTS,
                                          model = global_model, logger=None)

                    
                # w, loss , acc = local_model.update_weights(dataloader=dataloader, global_round=fold)
                # print("EPOCH",epoch)
                # print("TYPE EPOCH",type(epoch))
                w, loss , acc = local_model.update_weights(dataloader=dataloader, global_round=epoch)
                    

            
            
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                local_acc.append(copy.deepcopy(acc))
                global_model.load_state_dict(w)

                # print(f'Testing for Client {idx}')
                # accuracy, loss, y_true, y_pred
                # torch.cuda.empty_cache()
                client_acc, client_loss, y_true, y_pred, y_t, y_p = test_inference(global_model, test_loader)
                



###############################################################################################################

#PYCM SECTION
###############################################################################################################
                cm = pycm.ConfusionMatrix(y_t, y_p, digit = 5)
                # class_label_names = {k:v for k,v in zip (range(0,len(train_set[0].classes)), train_set[0].classes)}
                # cm.relabel(mapping = class_label_names)
                cm.stat(summary = True)
                to_save_as_a_file = cm.to_array()
                # with open(f'fold_{int(fold)}_client_{idx}_confusion_matris.csv', 'w') as write_file:
                    # write_file.write(to_save_as_a_file)
                # hpc
                cm.save_csv(os.path.join('skewed_results','local_results',f"fold_{int(fold)}_client_{idx}_summary"))
                # my laptop
                # cm.save_csv(os.path.join("..","results","federated_skewed","local", f"fold_{int(fold)}_client_{idx}_summary"), summary = True, matrix_save = False)

                confusion_matrix_df = pd.DataFrame(cm.table)
                confusion_matrix_df.to_csv(f'fold_{int(fold)}_client_{idx}_confusion_matrix.csv')
                truth_vs_preds_df = pd.DataFrame({'y_true': y_t, 'y_pred': y_p})
                # hpc
                truth_vs_preds_df.to_csv(os.path.join('skewed_results','local_results',f'fold_{int(fold)}_client_{idx}_truth_v_preds.csv'))
                # my laptop
                # truth_vs_preds_df.to_csv(os.path.join("..","results","federated_skewed","local", f'fold_{int(fold)}_client_{idx}_truth_v_preds.csv'))

                # cm.plot(cmap = plt.cm.Greens,
                    # number_label = True, 
                    # plot_lib = "matplotlib")
                # hpc
                # plt.savefig(os.path.join('skewed_results','local_results', f"fold_{int(fold)}_client_{idx}_cmplot_output.png"), facecolor = 'y', bbox_inches = "tight",pad_inches = 0.3, transparent = True)      
                # laptop
                # plt.savefig(os.path.join("..","results","federated_skewed","local", f"fold_{int(fold)}_client_{idx}_cmplot_output.png"), facecolor = 'y', bbox_inches = "tight",pad_inches = 0.3, transparent = True)

                

###############################################################################################################

# END PYCM SECTION
###############################################################################################################


                
                # print(f'| Y_True {y_true} | Y_Pred {y_pred} | Y_Preds: {y_preds}')
                
                history_c['client_acc'].append(client_acc)
                history_c['client_loss'].append(client_loss)


            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)
            # torch.cuda.empty_cache()
            
            if epoch != 0: # was == 1
            
                # torch.save(global_model.state_dict(), f'../save_new/fed_models/{global_model._get_name()}_E{epoch}_F{fold}.pth')
                AVG_WEIGHTS.append(global_model.state_dict())

            loss_avg = sum(local_losses) / len(local_losses)
            acc_avg = sum(local_acc) / len(local_acc)
            # acc_avg = sum
            train_loss.append(loss_avg)
            train_accuracy.append(acc_avg)

            # if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {fold} Folds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))








        # Test inference after completion of training
        # torch.cuda.empty_cache()
        test_acc, test_loss, y_true, y_pred, y_t, y_p = test_inference(global_model, test_loader)
        
        f_l = [f1_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
        p_l = [precision_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
        r_l = [recall_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
        # auc_l = [roc_auc_score(p.cpu().numpy(), t.cpu().numpy(), multi_class='ovr') for t,p in zip(y_true,y_pred)]

        f1_avg = sum(f_l)/len(f_l)
        # print('f1:', f1_s)
        p_avg = sum(p_l)/len(p_l)
        r_avg = sum(r_l)/len(r_l)
        
        # if not os.path.exists('../save_new/cf_matrix_final2'):
        #     os.mkdir('../save_new/cf_matrix_final2')
        # save_fig = f'../save_new/cf_matrix_final2/{global_model._get_name()}_fold_{fold}.png'
        # cf_fold = plot_confusion_matrix(cf_matrix, class_names=class_names)
        # cf_fold.savefig(save_fig)
                
        
        # with open('../save_new/metrics_per_fold.txt','a+') as f:
        #             f.write(f'Fold: {str(fold)}\tF1-Score: {round(f1_avg, 2)}\tPrecision: {str(round(p_avg,2))}\tRecall: {str(round(r_avg,2))}\tAccuracy: {str(round(test_acc,2))}\tLoss: {str(round(test_loss,2))}\n')
                    
                    
        # y_t = torch.tensor(y_true).to(device)
        # f1 = f1_score(probs.detach(), y_t)
        # print('f1 in function: ',f1)
        # p,r = precision_recall(probs.detach().to(device), y_t, average='weighted', num_classes=9)
        # print('p,r: ',p,r)
        
        # fold_cf_matrix = confusion_matrix(y_true, y_pred)

        # cf_fold=plot_confusion_matrix(fold_cf_matrix, class_names)
        
        # f1_avg, p_avg, r_avg = get_metrics(y_true, probs)
#         print('='*25, 'Metrics','='*25)
#         print(f'| F1-Score: {f1_avg} | Precision: {p_avg} | Recall: {r_avg} | AUC: {auc_avg} |')
#         print('='*100)
                
        # save_fig = f'../save_new/fed_cm/{global_model._get_name()}_fold_{fold}.png'
        # cf_fold.savefig(save_fig)
        # np.save(f'../save_new/fed_cm/{global_model._get_name()}_fold_{fold}.npy', fold_cf_matrix)



###############################################################################################################

#PYCM SECTION INFERENCE AFTER TRAINING
###############################################################################################################
        cm = pycm.ConfusionMatrix(y_t, y_p, digit = 5)
        # class_label_names = {k:v for k,v in zip (range(0,len(train_set[0].classes)), train_set[0].classes)}
        # cm.relabel(mapping = class_label_names)
        cm.stat(summary = True)
        to_save_as_a_file = cm.to_array()

        cm.save_csv(os.path.join('skewed_results','global_results',f"fold_{int(fold)}_client_{idx}_stats_summary"))
        # with open(f'fold_{int(fold)}_client_{idx}_confusion_matris.csv', 'w') as write_file:
            # write_file.write(to_save_as_a_file)
        # hpc
        # truth_vs_preds_df.to_csv(os.path.join('skewed_results','global_results',f"fold_{int(fold)}_client_{idx}_summary_global"))
        # my laptop
        # cm.save_csv(os.path.join("..","results","federated_skewed","global", f"fold_{int(fold)}_client_{idx}_summary_global"), summary = True, matrix_save = False)

        # confusion_matrix_df = pd.DataFrame(cm.table)
        # hpc
        # truth_vs_preds_df.to_csv(os.path.join('skewed_results','global_results',f'fold_{int(fold)}_client_{idx}_confusion_matrix_global.csv'))
        # my laptop
        # confusion_matrix_df.to_csv(os.path.join("..","results","federated_skewed","global", f'fold_{int(fold)}_client_{idx}_confusion_matrix_global.csv'))
        truth_vs_preds_df = pd.DataFrame({'y_true': y_t, 'y_pred': y_p})
        # hpc
        truth_vs_preds_df.to_csv(os.path.join('skewed_results','global_results',f'fold_{int(fold)}_client_{idx}_truth_v_preds_global.csv'))
        # my laptop
        # truth_vs_preds_df.to_csv(os.path.join("..","results","federated_skewed","global", f'fold_{int(fold)}_client_{idx}_truth_v_preds_global.csv'))

        # cm.plot(cmap = plt.cm.Greens,
            # number_label = True, 
            # plot_lib = "matplotlib")
        # hpc
        # plt.savefig(os.path.join('skewed_results','global_results', f"fold_{int(fold)}_client_{idx}_cmplot_output.png"), 
        #     facecolor = 'y', 
        #     bbox_inches = "tight",
        #     pad_inches = 0.3, 
        #     transparent = True)
        # laptop
        # plt.savefig(os.path.join("..","results","federated_skewed","global", f"fold_{int(fold)}_client_{idx}_cmplot_output.png"), 
            # facecolor = 'y', 
            # bbox_inches = "tight",
            # pad_inches = 0.3, 
            # transparent = True)


###############################################################################################################

# END FINAL PYCM SECTION
###############################################################################################################



# ======================= Save Model ======================= #
        if test_acc > best_acc:
            best_acc = test_acc
            # best_model_wts = copy.deepcopy(global_model.state_dict())
            # torch.save(global_model.state_dict(), f'../save_new/fed_models/{global_model._get_name()}_{args.optimizer}.pth')


        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_accuracy)
        history['train_loss'].append(train_loss)


        # print(f' \n Results after {fold} Folds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))


    df = pd.DataFrame(history)
    
    dfc = pd.DataFrame(history_c)
    df.to_csv(os.path.join(f'{global_model._get_name()}_{args.optimizer}_{k}Folds_{args.epochs}GE_{args.local_ep}LE.csv'))
    dfc.to_csv(os.path.join(f'{global_model._get_name()}_{args.optimizer}_{k}Folds_{args.epochs}GE_{args.local_ep}LE_Clients.csv'))
    
    FINAL_WEIGHTS = average_weights(AVG_WEIGHTS)
    GLOBAL_MODEL.load_state_dict(FINAL_WEIGHTS)
    # my laptop
    # torch.save(GLOBAL_MODEL.state_dict(),(os.path.join('..','results','federated_skewed',f'{global_model._get_name()}_{args.optimizer}_FINAL_WEIGHTS.pth'))
    # hpc
    torch.save(GLOBAL_MODEL.state_dict(),(os.path.join('skewed_results','global_results',f'{global_model._get_name()}_{args.optimizer}_FINAL_WEIGHTS.pth')))
    # kaggle
    # torch.save(GLOBAL_MODEL.state_dict(),(os.path.join(f'{global_model._get_name()}_{args.optimizer}_FINAL_WEIGHTS.pth'))
     
    

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))