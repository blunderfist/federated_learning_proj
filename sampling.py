#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms



def skin_cancer_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users




def skew_iid(dataset, num_users):
# splits is based on num of samples from each class for each user
    splits=[[184, 43, 37, 35, 18, 22, 19, 16, 13],
            [47, 176, 37, 35, 18, 22, 18, 15, 12],
            [46, 43, 150, 36, 18, 22, 18, 15, 12],
            [46, 44, 38, 143, 18, 22, 18, 15, 12],
            [46, 44, 38, 36, 72, 23, 18, 15, 12]]
    # added_files=[]
    user_groups={i:[] for i in range(num_users)}
    c = dataset.__distribution__()

    # all_fs=glob.glob('../../skin_cancer_data/train/**/*.jpg')
    # f=0.0
    for user in range(num_users):
        user_files=[]
        u_idxs=[]

        for j in range(len(c)):

            k = 9-(j+1)

            f = splits[user][j]
            # f_name = 
            # print(f[0])

            user_files = c[k]['files'][:f]
            f_names = [f.split('/')[-1].split('.')[0] for f in user_files]
            nms_ids = {i:v for v,i in enumerate(dataset.file_names)}
            idxs = set(nms_ids[f] for f in f_names)
            u_idxs.append(idxs)

            # print(user, idxs)

            user_groups[user].append(idxs)
            s =  set().union(*user_groups[user])
            user_groups[user]

            del c[k]['files'][:f]
            
    dict_users = combine_sets(user_groups, num_users)
    
    return dict_users


def combine_sets(u_dict, num_users):
    u_g={}

    for u in range(num_users):

        s = set().union(*u_dict[u])
        u_g[u] = s
        
    return u_g


