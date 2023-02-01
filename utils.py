#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os, sys
import torchvision

import copy
import os
import torch
from torchvision import datasets, transforms

from sampling import skin_cancer_iid, skew_iid

from skin_cancer_dataset import SkinCancer

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        raise('cifar dataset not implemented!')
  
    elif args.dataset == 'skin_cancer':
        data_dir = '../../skin_cancer_data_fed'
        
        dataset = SkinCancer(os.path.join(data_dir,'train'), transform=None)
        
        # val_dataset = SkinCancer(os.path.join(data_dir,'val'), transform=None)
        
        test_dataset = SkinCancer(os.path.join(data_dir,'test'), transform=None)
        
        # dataset = ConcatDataset([train_dataset,val_dataset])
        
        dataset_size = len(dataset)
        

#         if not args.federated:
#             data_dir = '../../skin_cancer_data_fed'

#             image_datasets = {x: SkinCancer(os.path.join(data_dir, x),
#                                                       transform=None)
#                               for x in ['train','test']}
#             dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
#                                                          shuffle=True, num_workers=4)
#                           for x in ['train', 'test']}
#             dataset_sizes = {x: len(image_datasets[x]) for x in ['train','test']}
        
#         elif args.federated:
#             data_dir = '../../skin_cancer_data_fed'

#             image_datasets = {x: SkinCancer(os.path.join(data_dir, x),
#                                                       transform=None)
#                               for x in ['train', 'val','test']}
#             dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
#                                                          shuffle=True, num_workers=4)
#                           for x in ['train', 'val','test']}
#             dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    
    
        # train_dataset = image_datasets['train']
        # test_dataset= image_datasets['val']
        
        if args.iid and not args.skew:
            # Sample IID user data from Mnist
            user_groups = skin_cancer_iid(dataset, args.num_users)
            
        elif args.iid and args.skew:
            user_groups = skew_iid(dataset, args.num_users)

    return dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
