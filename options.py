#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=3,
                        help="number of rounds of training")
    
    parser.add_argument('--num_users', type=int, default=5,
                        help="number of users: K")
    
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='custom_EN_b0_v2', help='model name')
   

    # other arguments
    parser.add_argument('--dataset', type=str, default='skin_cancer', help="name \
                        of dataset")
    
    
    parser.add_argument('--federated', type=bool, default=False, help="Use \
                        Federated Learning")
    
    
    parser.add_argument('--num_classes', type=int, default=9, help="number \
                        of classes")
    
    parser.add_argument('--gpu', default='mps', help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    
    parser.add_argument('--optimizer', type=str, default='adamx', help="type \
                        of optimizer")
    
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    
    parser.add_argument('--skew', type=bool, default=False,
                        help='Default set to IID. Set to 0 for non-IID.')
    
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    
 
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--danica_comp', type=bool, default=False, help='running on danicas computer device = mps')

    parser.add_argument('--freeze', type=bool, default=True, help='freeze layers for pretraining')

    args = parser.parse_args()
    
    return args
