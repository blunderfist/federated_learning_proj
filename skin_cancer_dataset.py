
import os, sys, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
import random
from do_augmentation import augment


class SkinCancer(Dataset):
    """Skin Cancer Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.transform = transform
        # self.task = task
        
        # if task == 'train':
        # self.data_dir = os.path.join(self.root_dir,self.task)
        
        self.class_paths = glob.glob(os.path.join(self.root_dir,'*'))
        # self.classes = [i.split('/')[-1] for i in self.class_paths]
        self.classes = [os.path.basename(i) for i in self.class_paths]
        
        self.class_id = {i:j for i, j in enumerate(self.classes)}
        self.class_to_id = {value:key for key,value in self.class_id.items()}
        
        self.files_paths = glob.glob(os.path.join(self.root_dir,'*/*.jpg'))
        # self.file_names = [f.split('/')[-1].split('.')[0] for f in self.files_paths]
        self.file_names = [os.path.basename(f).split('.')[0] for f in self.files_paths]
        self.file_names_ids = {i:v for v,i in enumerate(self.file_names)}


    def __len__(self):
        return len(glob.glob(self.root_dir+'/**/*.jpg'))
    
    
    def __distribution__(self):
        # # data_dir = self.root_dir + '/train'
        # classes_path = glob.glob(self.root_dir+'/**/*.jpg')
        # # classes_path = glob.glob('../../skin_cancer_data/Train'+'/*')
        # classes = [i.split('/')[-1] for i in classes_path]
        class_dcit_lists=[]
        for idx in range(0, len(self.classes)):
            class_dict = {}    
            class_dict['class'] = self.classes[idx]
            # class_dict['files'] = glob.glob(self.root_dir+'/'+self.classes[idx]+'/*.jpg')
            class_dict['files'] = glob.glob(os.path.join(self.root_dir, self.classes[idx], '*.jpg'))
            class_dict['size'] = len(class_dict['files'])
            class_dcit_lists.append(class_dict)
        sorted_list = sorted(class_dcit_lists, key= lambda class_dcit_lists: class_dcit_lists['size'])
        return sorted_list


    def __getitem__(self, idx):
        
        aug_list = [1,2,3,4,5,6,7]

        image_paths = glob.glob(self.root_dir + '/**/*.jpg')
        # image_paths = glob.glob(os.path.join(self.root_dir, '**', '*.jpg'))
        # print("IMAGE PATHS GLOB",image_paths)
        random.shuffle(image_paths)

        image = Image.open(image_paths[idx])
        # label = image_paths[idx].split('/')[-2] 
        label = os.path.basename(os.path.dirname(image_paths[idx]))
        # print(f"image_paths index = {image_paths[idx]}")
        # print(f"label = {label}")
        image = transforms.Resize(size=(224,224))(image)

        x = random.choice(aug_list)
        image_tensor = augment(image,x)
        # print(f"\n\n\n\n\nclass to id dict\n{self.class_to_id}")
        # print(f"\n\n\n\n\nclass to id dict\n{self.class_to_id['nevus']}")
        label_id = self.class_to_id[str(label)]
        return image_tensor, label_id
