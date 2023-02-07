import os
import sys
import copy
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import io
import datetime
import pickle

import torchmetrics
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SubsetRandomSampler, ConcatDataset

from options import args_parser
from _update import LocalUpdate
from utils import get_dataset, average_weights, exp_details
from skin_cancer_dataset import SkinCancer
import models
from models import EfficientNet, ResNet, VGG, custom_EN_b0, custom_EN_b0_v2#, custom_EN_b0_v3

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc

import pycm

import warnings
warnings.filterwarnings(action = 'ignore')
# warnings.filterwarnings('UndefinedMetricWarning')


def skin_cancer_iid_testing(dataset):
	"""
	Sample I.I.D. client data from MNIST dataset
	:param dataset:
	:return: dict of image index
	"""
	dict_users = {}
	dict_users[0] = [i for i in range(len(dataset))]

	return dict_users


def test_inference(model, testloader):
	
	""" Returns the test accuracy and loss.
	"""
	# torch.cuda.empty_cache()
	
	m = nn.Softmax(dim=1)
	
	loss, total, correct = 0.0, 0.0, 0.0
	# len(probs)
	if args.danica_comp:
		device = 'mps'
	else:
		device = 'cpu'
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




##########################
# Getting data
##########################


def build_test_set():
	"""builds list of all skewed datasets"""

	test_set = SkinCancer(os.path.join('..','data','Test'), transform = None)

	return test_set


# def get_test_set(client_idx):
# 	"""gets skincancer object for each client with path to their dataset"""

# 	test_set = SkinCancer(os.path.join('..','data','Test'), transform = None)

# 	return test_set


######################################################################################################################################
# Begin inference
######################################################################################################################################

if __name__ == '__main__':
	start_time = time.time()

	args = args_parser()
	exp_details(args)

	# added this so hopefully we aren't having to switch it constantly
	# Default set to danica's computer so don't worry about this argument
	# if I run on mine I'll set it to false, or if you run on another set --danica_comp False on cmd line
	if args.danica_comp:
		# device = 'mps'
		device = 'cpu'		
	else:
		device = 'cpu'

	# if args.gpu_id:
	#     torch.cuda.set_device(args.gpu_id)
	# try:
	#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#     # device = 'mps' if args.gpu else 'cpu'
	# except:
	#     device = 'cpu'
	# device = 'mps'
# 	device = 'cpu'


	# train_set, skewed_datasets = build_train_set_lst()
	test_set = build_test_set()

	# class_names =  [os.path.basename(i) for i in test_set[0].classes]
	# print(f"CLASS NAMES \n\n{class_names}\n\n")
	# test_set = SkinCancer(os.path.join('../../skin_cancer_data_fed','test'), transform=None)

	user_groups = skin_cancer_iid_testing(test_set)


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


	# this is changing the output if its not a custom model
	# our models have correct output size, pretrained do not
	custom = ['custom_EN_b0_v2', 'custom_EN_b0_v3']
	if args.model not in custom:
		old_fc = model.classifier.__getitem__(-1)
		new_fc = nn.Linear(in_features = old_fc.in_features, out_features = 9, bias = True)
		model.classifier.__setitem__(-1 , new_fc)

	# load weights
	# only model params are saved so they are loaded after model is loaded
	# path below should work for any model trained
	# os.path.join('skewed_results','global_results',f'{global_model._get_name()}_{args.optimizer}_FINAL_WEIGHTS.pth'))
	PATH = os.path.join('skewed_results','global_results',f'{model._get_name()}_{args.optimizer}_FINAL_WEIGHTS.pth')
	model.load_state_dict(torch.load(PATH))
	
	client_data = user_groups[0][:] # trying to copy all indexes from dataset for given client
	val_idx = client_data # test set

	test_sampler = SubsetRandomSampler(val_idx)
	
	# CAN THIS SUBSTITUTE AND WORK INSTEAD OF THIS CONVOUTED METHOD
	# test_set = SkinCancer(os.path.join('..','data','Test'), transform = None)
	
	test_set = torch.utils.data.Subset(test_set, test_sampler.indices)
	test_loader = DataLoader(test_set, batch_size=16,shuffle=False)


	# Test inference
	test_acc, test_loss, y_true, y_pred, y_t, y_p = test_inference(model, test_loader)
		


###############################################################################################################
# PYCM SECTION INFERENCE
###############################################################################################################

	if not os.path.exists(os.path.join('skewed_results','models',f'{model._get_name()}_{args.optimizer}_results','inference')):
		os.makedirs(os.path.join('skewed_results','models',f'{model._get_name()}_{args.optimizer}_results','inference'))

	cm = pycm.ConfusionMatrix(y_t, y_p, digit = 5)
	cm.stat(summary = True)
	to_save_as_a_file = cm.to_array()

	cm.save_csv(os.path.join('skewed_results','models',f'{model._get_name()}_{args.optimizer}_results','inference'))

	truth_vs_preds_df = pd.DataFrame({'y_true': y_t, 'y_pred': y_p})
	truth_vs_preds_df.to_csv(os.path.join('skewed_results','models',f'{model._get_name()}_{args.optimizer}_results','inference', 'truth_v_preds_global.csv'))


###############################################################################################################
# END PYCM SECTION
###############################################################################################################

	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
	print(f"\n\n{'*'*50}\nDONE\n{'*'*50}\n\n")