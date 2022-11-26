#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pandas as pd
import numpy as np
import os, glob, re

def get_avgs(lines):

    results_dict = {'F1-Score': [], 
                 'Precision': [], 
                 'Recall': [], 
                 'Accuracy': []}

    for line in lines:
        tmp = ' '.join(line.split())
        results = re.findall(r"\w*.\w*: [0-9]*.[0-9]*", tmp)

        for i, res in enumerate(results):
            item = res.strip().split(': ')
            if item[0] not in ['Fold', 'Loss']:
#                 print(item)
                results_dict[item[0]].append(item[1])
    return results_dict

def make_client_dict(res_dict):

    avgs_dict = {}
    for k,v in res_dict.items():
        avgs_dict[k] = sum(float(x) for x in v) / len(v)
    return avgs_dict

def combine(clients_dict):
    
    final_dict = {'F1-Score': [], 
                 'Precision': [], 
                 'Recall': [], 
                 'Accuracy': []}
    
    for client in clients_dict:
        for k,v in clients_dict[client].items():
            final_dict[k].append(v)
    return final_dict

def get_results():
    
    file_glob = glob.glob(os.path.join(os.getcwd(), '*.txt'))
    num_clients = len(file_glob)

    df_lst = []
    client_dict = {}
    for i, file in enumerate(file_glob):
        print(f"Opening {file}")
        with open(file, 'r') as read_file:
            lines = read_file.readlines()
            tmp_avgs = get_avgs(lines)
        client_dict["Client_"+str(i)] = make_client_dict(tmp_avgs)
#         print(client_dict)
        tmp_df = pd.DataFrame(client_dict.items(), columns = ["Metric", "Value"])
        df_lst.append(tmp_df)
    all_results = combine(client_dict)
    final = make_client_dict(all_results)
    df = pd.DataFrame(final.items(), columns = ["Metric", "Value"])
    df.to_csv("metrics_all_models_all_folds.csv", index = False)
    # return df#, df_lst
get_results()