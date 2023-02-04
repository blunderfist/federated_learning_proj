import os,sys
import copy
import time
# import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
# from skin_cancer_dataset import SkinCancer
import matplotlib.pyplot as plt
# import itertools
# import io

# from options import args_parser
# from _update import LocalUpdate
import datetime



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
		tmp_train_set =  SkinCancer(os.path.join('..','skewed_dataset',f'{skewed_datasets[i]}','train'), transform = None)
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
	train_set = SkinCancer(os.path.join('..','skewed_dataset',f'{skewed_datasets[client_idx]}','train'), transform = None)

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

	# args = args_parser()
	# exp_details(args)

	# if args.gpu_id:
	#     torch.cuda.set_device(args.gpu_id)
	# try:
	#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#     # device = 'mps' if args.gpu else 'cpu'
	# except:
	#     device = 'cpu'
# 	device = 'mps'
	device = 'cpu'
	# load dataset and user groups
	# train_dataset, test_dataset, user_groups = get_dataset(args)
	
	# train_set =  SkinCancer(os.path.join('../../skin_cancer_data_fed','train'), transform=None)
	# train_set =  SkinCancer(os.path.join(f'../../{d_?}','train'), transform=None)
#     ######################################################################################################################################

# ##################################
# changing the train set to be a list of training sets each from a skewed dataset
# these will be accessed by indexing in the kfold below
	# train_set, skewed_datasets = build_train_set_lst()
	# print(f"skewed_datasets{skewed_datasets}")
	# print("\n\n\n\n\n\n\n\nprinting training set so i know what it looks like")
	# for i in range(len(train_set)):
	#     print(train_set[i])
	#     print("len traingset",len(train_set),"\n\n\n\n\n\n\n\n\n")
##################################
	# tmp = train_set[0]
	# print("TRAIN SET",tmp.classes)



	# print("train set 0",train_set)
	# class_names =  [os.path.basename(i) for i in train_set[0].classes]
	# print(f"CLASS NAMES \n\n{class_names}\n\n")
	# test_set = SkinCancer(os.path.join('../../skin_cancer_data_fed','test'), transform=None)





	######################################################################################################################################
	
	# user_groups = skin_cancer_iid(train_set, 5) # was 2
	# user_groups = skin_cancer_skewed(train_set, 2)
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
	# if args.model == 'efficientnet':
	# 	pass


	# GLOBAL_MODEL = model

	# # this is changing the output if its not a custom model
	# # our models have correct output size, pretrained do not
	# custom = ['custom_EN_b0_v2', 'custom_EN_b0_v3']
	# if args.model not in custom:
	# 	old_fc = GLOBAL_MODEL.classifier.__getitem__(-1)
	# 	new_fc = nn.Linear(in_features=old_fc.in_features, out_features= 9, bias=True)
	# 	GLOBAL_MODEL.classifier.__setitem__(-1 , new_fc)
	# print(type(GLOBAL_MODEL))
	# # GLOBAL_MODEL.classifier[1].out_features = 9
	# # print(GLOBAL_MODEL.classifier)
	
	

	# # Set the model to train and send it to device.
	# GLOBAL_MODEL.to(device)
	# GLOBAL_MODEL.train()
	# # print(global_model)
	
	

	# # copy weights
	# GLOBAL_MODEL_WEIGHTS = copy.deepcopy(GLOBAL_MODEL.state_dict())
	
	
	# # global_weights = global_model.state_dict()

	# # Training
	# train_loss, train_accuracy = [], []
	
	# test_loss, test_accuracy = [], []
	
	# val_acc_list, net_list = [], []
	# cv_loss, cv_acc = [], []
	# print_every = 1
	# val_loss_pre, counter = 0, 0
	
	# history = {'train_loss': [], 'train_acc': [],
	# 		   'test_loss': [], 'test_acc': []}
	
	# history_c={'client_acc':[], 'client_loss':[]}
		
		
	# global_model = GLOBAL_MODEL

###############################################################################################################################    
	# k=2
	# k = 5
	# splits = KFold(n_splits = k, shuffle = True, random_state = 42)
	
# 	best_acc = 0.0
# 	AVG_WEIGHTS=[]
	


# ######################MOVIGN THIS TO A LOWER PORTION
# 	# for fold in range(k):##############################################################

# 		# This is new idea
# 		# now the kfold is just running through a range 0-k and using that as the fold #
# 		# the data is not being split here because it won't work with current setup
# 		# after a user_idx is selected below, data is split using a 4/5 1/5 split for train, val
# 		# currently missing function for this but will create


# 	local_weights=[]
# 	user_sets=[]
		
# 	global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
		
# 	# print(f'Model Initialized for fold {fold}...') # moved down into for loop just below

# 	k = 5 # K folds
# # ######################################################
# # ####    BEGIN KFOLD         
	# for fold in range(k):
# 		print(f'Model Initialized for fold {fold}...')
		
# ######################################################
		
		
	for epoch in tqdm(range(int(3))): # GLOBAL EPOCHS CURRENTLY AT 3
		# print(f'Model Initialized for fold {epoch}...')
		# local_weights, local_losses, local_acc = [], [], []
		print(f'\n | Global Training Round : {epoch+1} of 3|\n')



		# global_model.train()
		# m = max(int(args.frac * args.num_users), 1)
		m = max(int(1 * 5), 1)

		# pick a random client from the range of clients
		idxs_users = np.random.choice(range(5), m, replace=False)

			

		# for idx in idxs_users:
		for idx in range(5):

			print(f'User Index__________{idx} __________')



			k = 5 # K folds
# ######################################################
# ####    BEGIN KFOLD         
			for fold in range(k):
				print(f'Model Initialized for fold {fold}...')





				if epoch == 0:
					print("epoch should be = 0: is", epoch)
					# global_model.load_state_dict(GLOBAL_MODEL_WEIGHTS)
					# local_model = LocalUpdate(args=args, dataset=train_dataset,
					# 					  idxs=client_data, weights = GLOBAL_MODEL_WEIGHTS,
					# 					  model = global_model, logger=None)                    


				# wrong spot updates after each fold
				# elif epoch != 0: # was == 1
				# 	print("Averaging weights")				
				else:
					print("epoch should be > 0: epoch is", epoch)



			
			if epoch != 0: # was == 1
				print("Averaging weights")

		if epoch == 0: # if the first global epoch is finished we need to update weights before next epoch
			print("Averaging weights after first global epoch")
			print("This should only be printed ONCE")


	 
	

	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))