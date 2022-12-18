import os
from glob import glob
import numpy as np
import shutil


def del_dir_if_exist():
	"""deletes directories and all contents if they already exist"""
	skewed_data_dir = r'skewed_dataset'
	# deletes existing skewed data directory if it exists
	if os.path.exists(os.path.join(os.getcwd(), skewed_data_dir)):
		shutil.rmtree(os.path.join(os.getcwd(), skewed_data_dir))


def create_client_dirs(dir_glob, num_clients = 5):
	"""
	creating num_clients number of clients and creating directories for each with subdirectories for each class in the dataset
	args:
		dir_glob: directory to pull class names from
		num_clients: int, how many clients
	returns;
		None, creates new empty directories to be filled later
	"""

	clients = range(1, num_clients + 1)
	for client in clients:
		for directory in dir_glob:
#             os.mkdir(os.path.join('skewed_dataset',f'client_{client}',os.path.basename(os.path.normpath(directory))))
			print("Created _______", os.path.join('skewed_dataset',f'client_{client}',os.path.basename(os.path.normpath(directory))))
	print("All directories created succesfully\n")


def skew_and_distribute_imgs(dir_glob, num_clients):
	"""randomly selects classes to skew based on number of clients
	selects classes, skews data in favor of 50% to single client, evenly distributes the rest between remaining clients
	args:
		num_clients: number of clients
	returns:
		None, adds images to folders based on skewing decisions
		Select num_clients
		Randomly select same # of classes
			Shuffle class and assign 50% to one client
			Split remainder of class evenly between remaining clients
		For remaining unselected classes
			Shuffle, split evenly across all clients
	"""

	selected_classes_to_skew = np.random.choice(dir_glob, num_clients, replace = False)
	remaining_classes = [x for x in dir_glob if x not in selected_classes_to_skew]

	lst_of_lst_of_files_to_skew = []
	# list of list of filenames for each class to be skewed
	print(f"\n{'#'*15} Classes to be skewed between clients {'#'*15}\n")

	for i in selected_classes_to_skew:
		print(f'Class {os.path.basename(os.path.dirname(i))} has {len(i)} images')
		lst_of_lst_of_files_to_skew.append(glob(os.path.join(i, '*')))
		# print(glob(os.path.join(i, '*')))

	print(f"\n{'#'*15} Adding skewed classes to clients {'#'*15}\n")

	for i in range(len(lst_of_lst_of_files_to_skew)):
		# print([x for x in range(len(lst_of_lst_of_files_to_skew[i]))])
		idx_of_class = [x for x in range(len(lst_of_lst_of_files_to_skew[i]))]
		np.random.shuffle(idx_of_class)
		
		half_of_len_class = len(idx_of_class) // 2
		half_of_class = idx_of_class[:half_of_len_class]
		rest_of_class = idx_of_class[half_of_len_class:]
		
		for_skewed = [x for x in lst_of_lst_of_files_to_skew[i] if lst_of_lst_of_files_to_skew[i].index(x) in half_of_class]
		not_for_skewed = [x for x in lst_of_lst_of_files_to_skew[i] if lst_of_lst_of_files_to_skew[i].index(x) in rest_of_class]
		
	    # copies half of one class to one client
		for img in for_skewed:
			class_name = os.path.basename(os.path.dirname(img))
			# print(img)
			# print(os.path.join(os.getcwd(),'skewed_dataset',f'client_{i}','train',class_name,os.path.basename(img)))
			# dst = os.path.join(os.getcwd(),'skewed_dataset',f'client_{i}','train',class_name,os.path.basename(img))
			# shutil.copy(img, dst)
		print(f"Skewing {class_name}")
		print(f"Client {i}:\tAdded {len(for_skewed)} images to {class_name}\n")
   
		# copies rest of that class to each client equally
		idx_of_remaining = [x for x in range(len(not_for_skewed))]
		
		if len(idx_of_remaining) % (num_clients - 1) == 0:
			even_split = len(idx_of_remaining) // (num_clients - 1)
			last_files = 0
		else:
			even_split = len(idx_of_remaining) // (num_clients - 1)
			last_files = len(idx_of_remaining) % (num_clients - 1)
		start_idx = 0
		end_idx = 0
		remaining_clients = list(range(num_clients))
		remaining_clients.pop(i)
		print(f"Evenly splitting remainder of {class_name} to remaining clients")
		for split in remaining_clients:
			if split != remaining_clients[-1]:
				# print(split)
				other_idx = not_for_skewed[start_idx:end_idx + even_split]
				for img in other_idx:
					class_name = os.path.basename(os.path.dirname(img))
					# print(img)
					# print(os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img)))
					# dst = os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img))
					# shutil.copy(img, dst)
				print(f"Client {split}:\tAdded {len(other_idx)} images to {class_name}")
				start_idx += even_split
				end_idx += even_split
			else:
				other_idx = not_for_skewed[start_idx:end_idx + even_split + last_files]
				for img in other_idx:
					class_name = os.path.basename(os.path.dirname(img))
					# print(img)
					# print(os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img)))
					# dst = os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img))
					# shutil.copy(img, dst)
				print(f"Client {split}:\tAdded {len(other_idx)} images to {class_name}")
				print('\n')



	# remaining classes to be evenly split

	lst_of_lst_of_files_not_to_skew = []
	# list of list of filenames for each class to be skewed
	print(f"\n{'#'*15} Remaining classes to be evenly split between clients {'#'*15}\n")

	for i in remaining_classes:
		print(f'Class {os.path.basename(os.path.dirname(i))} has {len(i)} images')
		lst_of_lst_of_files_not_to_skew.append(glob(os.path.join(i, '*')))
	print(f"\n{'#'*15} Adding remaining classes evenly split between clients {'#'*15}\n")

	for i in range(len(lst_of_lst_of_files_not_to_skew)):
		idx_of_class = [x for x in range(len(lst_of_lst_of_files_not_to_skew[i]))]
		np.random.shuffle(idx_of_class)    
		even_split_class = [x for x in lst_of_lst_of_files_not_to_skew[i]]
		
		if len(even_split_class) % num_clients == 0:
			even_split = len(idx_of_class) // num_clients
		else:
			even_split = len(even_split_class) // num_clients
			last_files = len(even_split_class) % num_clients
		start_idx = 0
		end_idx = 0
		remaining_clients = list(range(num_clients))
		# print(f"{'#'*15}Evenly splitting {} between clients{'#'*15}")
		for split in range(num_clients):
			if split != (num_clients-1):
				other_idx = even_split_class[start_idx:end_idx + even_split]
				for img in other_idx:
					class_name = os.path.basename(os.path.dirname(img))
	                # print(os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img)))
					# dst = os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img))
					# shutil.copy(img, dst)
				print(f"Client {split}:\tAdded {len(other_idx)} images to {class_name}")
				start_idx += even_split
				end_idx += even_split

			else:
				other_idx = even_split_class[start_idx:end_idx + even_split + last_files]
				for img in other_idx:
					class_name = os.path.basename(os.path.dirname(img))
					# print(os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img)))
					# dst = os.path.join(os.getcwd(),'skewed_dataset',f'client_{split}','train',class_name,os.path.basename(img))
					# shutil.copy(img, dst)
				print(f"Client {split}:\tAdded {len(other_idx)} images to {class_name}")
				print('\n')



def main():
	"""Checks if directories exist, deletes if true
	Creates new empty directories
	Skews and adds data according to specifications"""

	# this needs to be removed when finalized, just for temp working with small dataset
	data_dir = os.path.join(os.getcwd(), 'd_1', 'train')
	# data_dir = os.path.join(os.getcwd(), 'data', 'train') # keep this one when name is finalized
	dir_glob = glob(os.path.join(data_dir, '*/'))

	# num_clients = int(input("Enter number of clients: "))
	num_clients = 5 # hardcoding for simplicity
	print("Deleting directories if they exist")
	# del_dir_if_exist() # uncomment this when deploying
	print("Creating directories...\n")
	create_client_dirs(dir_glob, num_clients)
	print("Directories created\n")
	skew_and_distribute_imgs(dir_glob, num_clients)
	print("\n\nSkewed dataset created\n\n")

if __name__ == '__main__':
	main()