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
from models import EfficientNet, ResNet, VGG, custom_EN_b0, custom_EN_b0_v2
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
#     df = pd.DataFrame(columns=['true', 'prob']) ###########################took this out
   
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
        # color = "white" if cm[i, j] < threshold else "black"
        plt.text(j, i, cm[i, j], 
            horizontalalignment="center", 
            # color=color,
            fontsize=7)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

    # return cf_matrix
    




    ######################################################################################################################################

def build_train_set_lst():
    """builds list of all skewed datasets"""
    skewed_datasets = ['d_1','d_2','d_3','d_4','d_5']
    train_set_lst = []
    for i in range(len(skewed_datasets)):
        # use this first one on D's computer
        # tmp_train_set =  SkinCancer(os.path.join(f'../../{skewed_datasets[i]}','train'), transform = None)
        # work comp
        tmp_train_set =  SkinCancer(os.path.join('data_skewed', f'{skewed_datasets[i]}','train'), transform = None)
        # my laptop
        # tmp_train_set =  SkinCancer(os.path.join('..',f'{skewed_datasets[i]}','train'), transform = None)
        train_set_lst.append(tmp_train_set)
    return train_set_lst
##################################


def skin_cancer_skewed(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
#######################ADD APRINT STATEMENT SO WE KNOW WHAT D_? IS WHO
    # randomly shuffle skewed datasets
    # print("datasets",dataset)
    np.random.shuffle(dataset)
    # assign randomly shuffled datasets to users in dictionary
    # num users must = number of datasets
    dict_users = {k:v for k,v in zip(range(num_users), dataset)}
    # num_items = int(len(dataset)/num_users)
    # arr = np.array([])
    # dict_users = {i: [] for i in range(5)}
    # dataset =  SkinCancer(os.path.join(f'../../{d_?}','train'), transform=None)
    # all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        tmp = []
        user_dataset =  dict_users[i]
        # print(i, user_dataset)
        # ch = np.random.choice(all_idxs, num_items, replace = False)
        all_idxs = [i for i in range(len(user_dataset))]
        # print("all_idxs",np.asarray(all_idxs))
        dict_users[i] = np.asarray(all_idxs)

    return dict_users


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
    device = 'cpu'

    # load dataset and user groups
    # train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # train_set =  SkinCancer(os.path.join('../../skin_cancer_data_fed','train'), transform=None)
    # train_set =  SkinCancer(os.path.join(f'../../{d_?}','train'), transform=None)
#     ######################################################################################################################################

# ##################################
# changing the train set to be a list of training sets each from a skewed dataset
# these will be accessed by indexing in the kfold below
    train_set = build_train_set_lst()
##################################
    # tmp = train_set[0]
    # print("TRAIN SET",tmp.classes)




    # class_names =  [i.split('/')[-1] for i in train_set.classes]
    class_names =  [os.path.basename(i) for i in train_set[0].classes]
    # print(class_names)
    # test_set = SkinCancer(os.path.join('../../skin_cancer_data_fed','test'), transform=None)






# def skin_cancer_skewed(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     skewed_datasets = ['d_1','d_2','d_3','d_4','d_5']
#     # create random generator
#     rand_choice = np.random.default_rng()
#     # randomly shuffle skewed datasets
#     random_datasets = rand_choice.choice(skewed_datasets, len(skewed_datasets), replace = False)
#     # assign randomly shuffled datasets to users in dictionary
#     dict_users = {i: random_datasets[I] for i in range(num_users)}
#     # num_items = int(len(dataset)/num_users)
#     # arr = np.array([])
#     dict_users = {i: [] for i in range(5)}
#     dataset =  SkinCancer(os.path.join(f'../../{d_?}','train'), transform=None)
#     # all_idxs = [i for i in range(len(dataset))]
#     for i in range(num_users):
#         user_dataset =  SkinCancer(os.path.join(f'../../{dict_users[i]}','train'), transform=None)
#         # ch = np.random.choice(all_idxs, num_items, replace = False)
#         all_idxs = [i for i in range(len(user_dataset))]
#         dict_users[i].append(ch)
#         dict_users[i] = np.asarray(ch)
#         # all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users






    ######################################################################################################################################
    
    # user_groups = skin_cancer_iid(train_set, 5) # was 2
    user_groups = skin_cancer_skewed(train_set, 5)

    # print(f"\n\n\nuser groups\n\n{user_groups}\n\n\n\n")
    ######################################################################################################################################









    

    GLOBAL_MODEL = torchvision.models.efficientnet_b0(pretrained=True)
    # GLOBAL_MODEL = custom_EN_b0_v2(pretrained=True)
    
    old_fc = GLOBAL_MODEL.classifier.__getitem__(-1)
    new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 9, bias=True)
    GLOBAL_MODEL.classifier.__setitem__(-1 , new_fc)
    
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
    k = 5
    splits = KFold(n_splits = k, shuffle = True, random_state = 42)
    
    best_acc = 0.0
    AVG_WEIGHTS=[]
    
    # GLOBAL_W=[]
    
    # auroc = torchmetrics.AUROC(num_classes=9)
#     f1_score = torchmetrics.F1Score(num_classes=9,average ='weighted')
    

################################################################################################################################
# added [data_index] here to only access the d_1 - d_5 dataset form the list of train_set    
    data_index = 0
    # print("LEN OF TRAIN SET DATA INDEX",len(train_set[data_index]))
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_set[data_index])))):
    
        # print("LEN OF TRAIN train_sampler train_idx",train_idx)
        # print("LEN OF TRAIN train_sampler val_idx",val_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

################################################################################################################################
# added [data_index] here too
        train_dataset = torch.utils.data.Subset(train_set[data_index], train_sampler.indices)
        test_set = torch.utils.data.Subset(train_set[data_index], test_sampler.indices)



        test_loader = DataLoader(test_set, batch_size=16,shuffle=False)
################################################################################################################################

################################################################################################################################
        # dict_users = skin_cancer_iid(train_set, 2)
        # dict_users = skin_cancer_iid(train_set, 5)

        # train_loaders=[]

        local_weights=[]
        user_sets=[]
        
        global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
        
        print(f'Model Initialized for client {fold}...')
        
        
        for epoch in tqdm(range(2)):
            local_weights, local_losses, local_acc = [], [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        

        
        
            local_weights, local_losses, local_acc = [], [], []
            # print(f'\n | Global Training Round : {fold+1} |\n')

            # global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                print(f'User Index__________{idx} __________')
                
         

######################################################
# added idx here data_index
                # print("\n\n\n\n\n\n\n\ntrain_set[idx]",train_set[data_index])
                dataloader = DataLoader(train_set[data_index], batch_size=16, sampler=train_sampler)
                # local_model = LocalUpdate(args=args, dataset=train_dataset,
                #                           idxs=user_groups[idx],global_weights = GLOBAL_MODEL_WEIGHTS,
                #                           model =GLOBAL_MODEL, logger=None)
                
                
                if epoch == 0:
                    global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], weights = GLOBAL_MODEL_WEIGHTS,
                                          model = global_model, logger=None)
                    
                    
                
                    # w, loss , acc= local_model.update_weights(
                    #     model=copy.deepcopy(GLOBAL_MODEL), dataloader=dataloader, global_round=fold)
                
                else:
                    global_model.load_state_dict(global_weights)
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx] , weights = GLOBAL_MODEL_WEIGHTS,
                                          model = global_model, logger=None)
                    
                w, loss , acc = local_model.update_weights(dataloader=dataloader, global_round=fold)
                    
                    
                
                # w, loss , acc= local_model.update_weights(
                #         model=GLOBAL_MODEL, dataloader=dataloader, global_round=fold)
            
            
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
                # cm = pycm.ConfusionMatrix(matrix={"Class1": {"Class1": 1, "Class2":2}, "Class2": {"Class1": 0, "Class2": 5}})
                class_label_names = {k:v for k,v in zip (range(0,len(train_set[0].classes)), train_set[0].classes)}
                cm.relabel(mapping = class_label_names)
                cm.stat(summary = True)
                to_save_as_a_file = cm.to_array()
                # with open(f'fold_{int(fold)}_client_{idx}_confusion_matris.csv', 'w') as write_file:
                    # write_file.write(to_save_as_a_file)
                cm.save_csv(os.path.join('skewed_results','local_results',f"fold_{int(fold)}_client_{idx}_summary"), summary = True, matrix_save = False)

                confusion_matrix_df = pd.DataFrame(cm.table)
                confusion_matrix_df.to_csv(os.path.join('skewed_results','local_results',f'fold_{int(fold)}_client_{idx}_confusion_matrix.csv'))
                truth_vs_preds_df = pd.DataFrame({'y_true': y_t, 'y_pred': y_p})
                truth_vs_preds_df.to_csv(os.path.join('skewed_results','local_results',f'fold_{int(fold)}_client_{idx}_truth_v_preds.csv'))

                cm.plot(cmap = plt.cm.Greens,
                    number_label = True, 
                    plot_lib = "matplotlib")
                plt.savefig(os.path.join('skewed_results','local_results',f"fold_{int(fold)}_client_{idx}_cmplot_output.png"), 
                    facecolor = 'y', 
                    bbox_inches = "tight",
                    pad_inches = 0.3, 
                    transparent = True)



###############################################################################################################

# END PYCM SECTION
###############################################################################################################



                # f_l = [f1_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
                # p_l = [precision_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
                # r_l = [recall_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
                # # auc_l = [roc_auc_score(p.cpu().numpy(), t.cpu().numpy(), multi_class='ovr') for t,p in zip(y_true,y_pred)]

                # f1_avg = sum(f_l)/len(f_l)
                # # print('f1:', f1_s)
                # p_avg = sum(p_l)/len(p_l)
                # r_avg = sum(r_l)/len(r_l)
                
                # cf_matrix = confusion_matrix(np.asarray(y_t), np.asarray(y_p))
                # # auc_score = roc_auc_score(np.asarray(y_t), np.asarray(y_p),multi_class='ovr')
                # # print(f'AUC: {auc_score}')
                # ave_fig = f'../save_new/cf_matrix_final2/{global_model._get_name()}_fold_{fold}_epoch_{epoch}_client_{idx}.png'
                # # ave_fig = f'cf_matrix_final2/{global_model._get_name()}_fold_{fold}_epoch_{epoch}_client_{idx}.png'
                # cf_fig = plot_confusion_matrix(cf_matrix, class_names=class_names)
                # cf_fig.savefig(ave_fig)
    
                
                
                
                # print('='*25, 'Metrics Per Client','='*25)
                # print(f'| F1-Score: {f1_avg} | Precision: {p_avg} | Recall: {r_avg} | AUC: {auc_avg} |')
                # print('='*100)

                # time = str(datetime.datetime.now()).split('.')[0]
                # time = time.sreplace(":", "_")
                # if not os.path.exists('../save_new'):
                #     os.mkdir('../save_new')
                # with open('../save_new/metrics_per_client.txt','a+') as f:
                # # with open(f'metrics_per_client.txt','a+') as f:
                #     f.write(f'Client: {str(idx)}\tF1-Score: {round(f1_avg, 2)}\tPrecision: {str(round(p_avg,2))}\tRecall: {str(round(r_avg,2))}\tAccuracy: {str(round(client_acc,2))}\tLoss: {str(round(client_loss,2))}\n')
                    
        
                
                # print(f'| Y_True {y_true} | Y_Pred {y_pred} | Y_Preds: {y_preds}')
                
                history_c['client_acc'].append(client_acc)
                history_c['client_loss'].append(client_loss)

                
                # confusion_matrix(y_true, y_pred)
                
                ### Confusion Matrix Per Fold, Epoch, Client ###
                # cf_matrix = confusion_matrix(y_true, y_pred)
                # print(cf_matrix)
                
                # f1 = f1_score(probs[0].detach(), torch.tensor(y_true[0]))
                # print(f'| Fold {fold} | Client {idx} | Client Accuracy: {client_acc} | Client Loss: {client_loss} | F1-Score: {f1}')
                      
                # area_auc = roc_auc_score(y_true, y_preds, multi_class='ovr')
                
                
                # print(f'| Fold {fold} | Client {idx} | Client Accuracy: {client_acc} | Client Loss: {client_loss}')
                
                
                
                # cf_figure=plot_confusion_matrix(cf_matrix, class_names)
                # save_fig = f'../save_new/fed_cm/{global_model._get_name()}_fold_{fold}_epoch_{epoch}_client_{idx}.png'
                # cf_figure.savefig(save_fig)
                # np.save(f'../save_new/fed_cm/{global_model._get_name()}_fold_{fold}_epoch_{epoch}_client_{idx}.npy', cf_matrix)


                # print(f'Confusion Matrix | Client {idx} | Fold {fold}')


            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)
            # torch.cuda.empty_cache()
            
            if epoch == 1:
            
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
        # cm = pycm.ConfusionMatrix(matrix={"Class1": {"Class1": 1, "Class2":2}, "Class2": {"Class1": 0, "Class2": 5}})
        class_label_names = {k:v for k,v in zip (range(0,len(train_set[0].classes)), train_set[0].classes)}
        cm.relabel(mapping = class_label_names)
        cm.stat(summary = True)
        to_save_as_a_file = cm.to_array()
        # with open(f'fold_{int(fold)}_client_{idx}_confusion_matris.csv', 'w') as write_file:
            # write_file.write(to_save_as_a_file)
        cm.save_csv(os.path.join('skewed_results','global_results', f"fold_{int(fold)}_client_{idx}_summary_global"), summary = True, matrix_save = False)

        confusion_matrix_df = pd.DataFrame(cm.table)
        confusion_matrix_df.to_csv(os.path.join('skewed_results','global_results',f'fold_{int(fold)}_client_{idx}_confusion_matrix_global.csv'))
        truth_vs_preds_df = pd.DataFrame({'y_true': y_t, 'y_pred': y_p})
        truth_vs_preds_df.to_csv(os.path.join('skewed_results','global_results',f'fold_{int(fold)}_client_{idx}_truth_v_preds_global.csv'))

        cm.plot(cmap = plt.cm.Greens,
            number_label = True, 
            plot_lib = "matplotlib")
        plt.savefig(os.path.join('skewed_results','global_results', f"f'fold_{int(fold)}_client_{idx}_cmplot_output.png"), 
            facecolor = 'y', 
            bbox_inches = "tight",
            pad_inches = 0.3, 
            transparent = True)



###############################################################################################################

# END PYCM SECTION
###############################################################################################################

# ======================= Save Model ======================= #
        if test_acc > best_acc:
            best_acc = test_acc
            # best_model_wts = copy.deepcopy(global_model.state_dict())
            # torch.save(global_model.state_dict(), f'../save_new/fed_models/{global_model._get_name()}_{args.optimizer}.pth')


        # history['test_acc'].append(test_acc)
        # history['test_loss'].append(test_loss)
        # history['train_acc'].append(train_accuracy)
        # history['train_loss'].append(train_loss)


        # print(f' \n Results after {fold} Folds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    ############################################################################################################
    data_index += 1    
    ############################################################################################################





    df = pd.DataFrame(history)
    
    dfc = pd.DataFrame(history_c)
    # df.to_csv(f'../save_new/fed_csvs/{global_model._get_name()}_{args.optimizer}_{k}Folds_{args.epochs}GE_{args.local_ep}LE.csv')
    # dfc.to_csv(f'../save_new/fed_csvs/{global_model._get_name()}_{args.optimizer}_{k}Folds_{args.epochs}GE_{args.local_ep}LE_Clients.csv')
    
    FINAL_WEIGHTS = average_weights(AVG_WEIGHTS)
    GLOBAL_MODEL.load_state_dict(FINAL_WEIGHTS)
    # torch.save(GLOBAL_MODEL.state_dict(),f'../save_new/fed_models/{global_model._get_name()}_{args.optimizer}_FINAL_WEIGHTS.pth')
    
    

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))