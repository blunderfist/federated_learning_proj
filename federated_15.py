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

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc

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
	device = 'mps'
	# device = 'cpu'

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



# def get_metrics(y_true, probs):
# 	metrics = {'f1':[],
# 			   'precision':[],
# 			   'recall':[],
# 			   'auc':[]}
	
	
# 	# for i in range(len(y_true)):
	
# 	f_l = [f1_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
# 	p_l = [precision_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
# 	r_l = [recall_score(p.cpu().numpy(), t.cpu().numpy(), average='macro') for t,p in zip(y_true,y_pred)]
	
	
		
# 	metrics['f1'].append(sum(f_l)/len(f_l))
# 	# p,r = precision_recall(sum(f_l)/len(f_l))
# 	metrics['precision'].append(sum(p_l)/len(p_l))
# 	# metrics['recall'].append(r.numpy())
# 	metrics['recall'].append(sum(r_l)/len(r_l))
	
# 	df_m = pd.DataFrame(metrics)
	
# 	# f1_avg = df_m['f1'].mean()
# 	# p_avg = df_m['precision'].mean()
# 	# r_avg = df_m['recall'].mean()
# 	# auc_avg = df_m['auc'].mean()
	
# 	# print(f1_avg, p_avg, r_avg, auc_avg)
# 	return df_m


# def get_metrics(y_true, probs):
# 	# metrics = {'f1':[],
# 	#            'precision':[],
# 	#            'recall':[],
# 	#            'auc':[]}
# 	y_true = torch.tensor(y_true).to(device)
	
	
# 	# for i in range(len(y_true)):
		
# 	f1 = f1_score(probs.detach(), y_true)
# 	print('f1 in function: ',f1)
# 	p,r = precision_recall(probs.detach().to(device), y_true, average='weighted', num_classes=9)
# 	print('p,r: ',p,r)
# 	# metrics['precision'].append(p.numpy())
# 	# metrics['recall'].append(r.numpy())
# 	# auc = auroc(probs.detach().to(device), y_true[-1])    
# #     df_m = pd.DataFrame(metrics)
	
# #     f1_avg = df_m['f1'].mean()
# #     p_avg = df_m['precision'].mean()
# #     r_avg = df_m['recall'].mean()
# #     auc_avg = df_m['auc'].mean()
	
# 	# print(f1_avg, p_avg, r_avg, auc_avg)
# 	return f1, p,r




##########################
# Getting data for runs
##########################

def build_train_set_lst():
	"""builds list of all skewed datasets"""

	skewed_datasets = [f'client_{x}' for x in range(1, 6)]
	train_set_lst = []
	for i in range(len(skewed_datasets)):
		tmp_train_set =  SkinCancer(os.path.join('..','skewed_dataset',f'{skewed_datasets[i]}','train'), transform = None)
		train_set_lst.append(tmp_train_set)

	skewed_lst = {k:v for k,v in zip(range(len(skewed_datasets)), train_set_lst)}
	# print("skewed_lst",skewed_lst) # this will tell us what dataset is from which folder for sanity check
	return train_set_lst, skewed_lst


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

	skewed_datasets = [f'client_{x}' for x in range(1, 6)]
	train_set = SkinCancer(os.path.join('..','skewed_dataset',f'{skewed_datasets[client_idx]}','train'), transform = None)

	return train_set



###########################################################################################
# KFOLD CODE
# we may not need this now that the code has been reworked
# if possible we should switch to sklearn, probably more robust
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

# this was useful early on trying to figure out why data was loading wrong, probably can delete if runs are smooth
def sanity_check_datasets(clients, skewed_ds):
	"""maps clients to dataset folders for santity checking all datasets are being used"""

	print(f"clients:\t\t{clients}") # prints entire client folder indexes
	print(f"clients keys:\t\t{clients.keys()}") # prints just the keys
	print(f"skewed_ds:\t\t{skewed_ds}")


###########################################################################
# Makes directories to save output if they don't exist yet   
###########################################################################
if not os.path.exists(os.path.join('skewed_results','global_results')):
	os.makedirs(os.path.join('skewed_results','global_results'))
if not os.path.exists(os.path.join('skewed_results','local_results')):
	os.makedirs(os.path.join('skewed_results','local_results')) 


######################################################################################################################################
# Begin federated learning
######################################################################################################################################

if __name__ == '__main__':
	start_time = time.time()

	# revisit to delete this?
	# define paths
	# path_project = os.path.abspath('..')
	# logger = SummaryWriter('../logs')

	args = args_parser()
	exp_details(args)

	# added this so hopefully we aren't having to switch it constantly
	# Default set to danica's computer so don't worry about this argument
	# if I run on mine I'll set it to false, or if you run on another set --danica_comp False on cmd line
	if args.danica_comp:
		device = 'mps'
		device = 'cpu'		
	else:
		device = 'cpu'
	device = 'mps'

	# if args.gpu_id:
	#     torch.cuda.set_device(args.gpu_id)
	# try:
	#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#     # device = 'mps' if args.gpu else 'cpu'
	# except:
	#     device = 'cpu'
	# device = 'mps'
# 	device = 'cpu'

######################################################################################################################################
# changing the train set to be a list of training sets each from a skewed dataset
# these will be accessed by indexing in the kfold below
	train_set, skewed_datasets = build_train_set_lst()
	# print(f"skewed_datasets{skewed_datasets}")
	# print("\n\n\n\n\n\n\n\nprinting training set so i know what it looks like")
	# for i in range(len(train_set)):
	#     print(train_set[i])
	#     print("len traingset",len(train_set),"\n\n\n\n\n\n\n\n\n")
	# tmp = train_set[0]
	# print("TRAIN SET",tmp.classes)



	# print("train set 0",train_set)
	class_names =  [os.path.basename(i) for i in train_set[0].classes]
	# print(f"CLASS NAMES \n\n{class_names}\n\n")
	# test_set = SkinCancer(os.path.join('../../skin_cancer_data_fed','test'), transform=None)


######################################################################################################################################
# Clients get their data assigned
######################################################################################################################################
	
	# user_groups = skin_cancer_iid(train_set, 5) # was 2
	user_groups = skin_cancer_skewed(train_set, args.num_users)
	# print("user_groups type",type(user_groups))
	# print("user_groups",user_groups)
	# this is showing us which client has which dataset
	# not easy to tell which is which but we can determine each is accounted for
	# print(f"\n\nSkewed Dataset Ordering\n\n{[v for v in skewed_datasets.values()]}\n\n\n")
######################################################################################################################################

#######################################################################
# sanity_check_datasets(user_groups, skewed_datasets)
#######################################################################


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

	# elif args.model == 'custom_EN_b0_v3':
	# 	model = custom_EN_b0_v3(args)


###################################################################################################################
### FREEZING WEIGHTS HERE
### defaults to True
### disable freezing with --freeze False on cmd line
###################################################################################################################

	if args.freeze:
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
		new_fc = nn.Linear(in_features = old_fc.in_features, out_features = 9, bias = True)
		GLOBAL_MODEL.classifier.__setitem__(-1 , new_fc)
	
	# Set the model to train and send it to device.
	GLOBAL_MODEL.to(device)
	GLOBAL_MODEL.train()
	
	# copy weights
	GLOBAL_MODEL_WEIGHTS = copy.deepcopy(GLOBAL_MODEL.state_dict())
	
	# global_weights = global_model.state_dict()

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

	best_acc = 0.0
	AVG_WEIGHTS=[]

	local_weights=[]
	user_sets=[]
		
	global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
		
	# for saving results later
	if not os.path.exists(os.path.join('skewed_results','models',f'{args.model}_{args.optimizer}_results')):
		os.makedirs(os.path.join('skewed_results','models',f'{args.model}_{args.optimizer}_results'))
	model_path = os.path.join('skewed_results','models',f'{args.model}_{args.optimizer}_results')
######################################################
# logic
# for each global epoch
# 	for each client perform kfold cross validation
# at end of global epoch avg weights
# repeat until all global epochs complete
######################################################

	# START GLOBAL EPOCHS
	for epoch in tqdm(range(int(args.epochs))):
		local_weights, local_losses, local_acc = [], [], []
		print(f'\n | Global Training Round : {epoch+1} of {args.epochs}|\n')

		global_model.train()
		# honestly can't remember what this does, Danica can explain
		m = max(int(args.frac * args.num_users), 1)

		# pick a random client from the range of clients
		idxs_users = np.random.choice(range(args.num_users), m, replace=False)

		##################################################################
		# iterating through each client to perform kfold cv for this epoch
		##################################################################

		for idx in range(args.num_users):

			print(f'Client Index__________{idx+1} __________')


			######################################################
			####    BEGIN KFOLD FOR THIS CLIENT
			######################################################

			k = 5 # set K folds
			for fold in range(k):
				print(f'Model Initialized for fold {fold+1} of {k}')

				client_data = user_groups[idx][:] # trying to copy all indexes from dataset for given client
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
				# print(f"\n\nDATA LEN: {len(client_data)}\n\n") # can help distinguish between dif selections of data, should be dif for all clients
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
# 				# added [data_index] here too
				train_dataset = torch.utils.data.Subset(train_set, train_sampler.indices)
				test_set = torch.utils.data.Subset(train_set, test_sampler.indices)
				test_loader = DataLoader(test_set, batch_size=args.local_bs,shuffle=False)



				dataloader = DataLoader(train_set, batch_size=args.local_bs, sampler=train_sampler)
				print(f'Client Index__________{idx+1} __________')
				if epoch == 0:
					# print("epoch ==0: epoch is", epoch)
					global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
					local_model = LocalUpdate(args=args, dataset=train_dataset,
										  idxs=client_data, weights = GLOBAL_MODEL_WEIGHTS,
										  model = global_model, logger=None)                    
				
				else:
					# print("epoch > 0: epoch is", epoch)

					global_model.load_state_dict(global_weights)
					local_model = LocalUpdate(args=args, dataset=train_dataset,
										  idxs=client_data , weights = GLOBAL_MODEL_WEIGHTS,
										  model = global_model, logger=None)

				w, loss , acc = local_model.update_weights(dataloader=dataloader, global_round=epoch)
					
				local_weights.append(copy.deepcopy(w))
				local_losses.append(copy.deepcopy(loss))
				local_acc.append(copy.deepcopy(acc))
				global_model.load_state_dict(w)

				# print(f'Testing for Client {idx}')
				# accuracy, loss, y_true, y_pred
				# torch.cuda.empty_cache()
				client_acc, client_loss, y_true, y_pred, y_t, y_p = test_inference(global_model, test_loader)

				
				# # print(f'| Y_True {y_true} | Y_Pred {y_pred} | Y_Preds: {y_preds}')
				
				# history_c['client_acc'].append(client_acc)
				# history_c['client_loss'].append(client_loss)


		# avg all client weights
		global_weights = average_weights(local_weights)

		# update global weights loading into model
		global_model.load_state_dict(global_weights)
		# torch.cuda.empty_cache()
			
		# if epoch != 0: # was == 1
			# print("avg weights!!!: epoch is", epoch)
		if not os.path.exists(os.path.join('fed_models')):
			os.mkdir('fed_models')
		torch.save(global_model.state_dict(), os.path.join(f'fed_models',f'{args.model}_E{epoch}_F{fold}.pth'))
		AVG_WEIGHTS.append(global_model.state_dict())
		print(f"{'#'*20}\n\t\t\tAveraging weights\n{'#'*20}\n\n")
	# loss_avg = sum(local_losses) / len(local_losses)
	# acc_avg = sum(local_acc) / len(local_acc)
	# # acc_avg = sum
	# train_loss.append(loss_avg)
	# train_accuracy.append(acc_avg)

	# Test inference after completion of training
	# torch.cuda.empty_cache()
	test_acc, test_loss, y_true, y_pred, y_t, y_p = test_inference(global_model, test_loader)
		


###############################################################################################################
#PYCM SECTION INFERENCE AFTER TRAINING
###############################################################################################################

	cm = pycm.ConfusionMatrix(y_t, y_p, digit = 5)
	# class_label_names = {k:v for k,v in zip (range(0,len(train_set[0].classes)), train_set[0].classes)}
	# cm.relabel(mapping = class_label_names)
	cm.stat(summary = True)
	to_save_as_a_file = cm.to_array()

	cm.save_csv(os.path.join(model_path,f"fold_{int(fold)}_client_{idx}_stats_summary"))

	# confusion_matrix_df = pd.DataFrame(cm.table)
	truth_vs_preds_df = pd.DataFrame({'y_true': y_t, 'y_pred': y_p})
	truth_vs_preds_df.to_csv(os.path.join(model_path, f'fold_{int(fold)}_client_{idx}_truth_v_preds_global.csv'))

	# cm.plot(cmap = plt.cm.Greens,
		# number_label = True, 
		# plot_lib = "matplotlib")

	# plt.savefig(os.path.join("..","results","federated_skewed","global", f"fold_{int(fold)}_client_{idx}_cmplot_output.png"), 
		# facecolor = 'y', 
		# bbox_inches = "tight",
		# pad_inches = 0.3, 
		# transparent = True)


###############################################################################################################
# END FINAL PYCM SECTION
###############################################################################################################

	# model_path = os.path.join('skewed_results','models',f'{global_model._get_name()}_{args.optimizer}_results')

# ======================= Save Model ======================= #
	if test_acc > best_acc:
		best_acc = test_acc
		# best_model_wts = copy.deepcopy(global_model.state_dict())
		# torch.save(global_model.state_dict(), f'../save_new/fed_models/{global_model._get_name()}_{args.optimizer}.pth')


	history['test_acc'].append(test_acc)
	history['test_loss'].append(test_loss)
	history['train_acc'].append(train_accuracy)
	history['train_loss'].append(train_loss)

	df = pd.DataFrame(history)
	
	dfc = pd.DataFrame(history_c)
	df.to_csv(os.path.join(model_path, f'{args.model}_{args.optimizer}_{k}Folds_{args.epochs}GE_{args.local_ep}LE.csv'))
	dfc.to_csv(os.path.join(model_path, f'{args.model}_{args.optimizer}_{k}Folds_{args.epochs}GE_{args.local_ep}LE_Clients.csv'))
	
	FINAL_WEIGHTS = average_weights(AVG_WEIGHTS)
	GLOBAL_MODEL.load_state_dict(FINAL_WEIGHTS)


	torch.save(GLOBAL_MODEL.state_dict(),(os.path.join(model_path,f'{args.model}_{args.optimizer}_FINAL_WEIGHTS.pth')))
 
	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
	print(f"\n\n{'*'*50}\nDONE\n{'*'*50}\n\n")
