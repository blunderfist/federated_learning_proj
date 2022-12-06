
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
        self.file_names = [f.split('/')[-1].split('.')[0] for f in self.files_paths] 
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
            class_dict['files'] = glob.glob(self.root_dir+'/'+self.classes[idx]+'/*.jpg')
            class_dict['size'] = len(class_dict['files'])
            class_dcit_lists.append(class_dict)
        sorted_list = sorted(class_dcit_lists, key= lambda class_dcit_lists: class_dcit_lists['size'])
        return sorted_list


    def __getitem__(self, idx):
        # print("IDX",idx,"\n\n\n\n\n\n\n\n\n")
        aug_list = [1,2,3,4,5,6,7,8]
        
        image_paths = glob.glob(self.root_dir+'/**/*.jpg')
        # print("image_paths",image_paths,"\n\n\n\n\n\n\n\n\n")

        random.shuffle(image_paths)
        
        
        image = Image.open(image_paths[idx])
        # print("image",image,"\n\n\n\n\n\n\n\n\n")

        # label = image_paths[idx].split('/')[-2] 
        label = os.path.basename(os.path.split(image_paths[idx])[0])
        # print("label",label,"\n\n\n\n\n\n\n\n\n")
        
        image = transforms.Resize(size=(224,224))(image)

        x = random.choice(aug_list)
        image_tensor = augment(image,x)
        
        label_id = self.class_to_id[str(label)]
        return image_tensor, label_id





#################################### ADDED ###########################################

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h)
        left = torch.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        # landmarks = landmarks - [left, top]

        return {'image': image}#, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, landmarks = sample['image'], sample['landmarks']
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}#,
                # 'landmarks': torch.from_numpy(landmarks)}

