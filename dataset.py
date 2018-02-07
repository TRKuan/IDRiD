# -*- coding: utf-8 -*
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random

class IDRiD_sub1_dataset(Dataset):

    def __init__(self, root_dir, task_type):
        if task_type not in ['MA', 'EX', 'HE', 'SE']: 
            raise ValueError('No such task type "%s"'%(task_type))
            
        self.root_dir = root_dir
        self.data_idx = []#(image_dir, mask_dir, name), mask_dir = None for NAR images

        image_root = os.path.join(root_dir, 'Apparent Retinopathy')
        image_NAR_root = os.path.join(root_dir, 'No Apparent Retinopathy')
        mask_root = os.path.join(root_dir, task_type)
        
        #Get the file index
        #AR images
        for filename in os.listdir(mask_root):
            image_dir = os.path.join(image_root, filename[:-7]+'.jpg')
            mask_dir = os.path.join(mask_root, filename)
            name = filename[:-7]
            self.data_idx.append((image_dir, mask_dir, name))
        #NAR images
        for filename in os.listdir(image_NAR_root):
            image_dir = os.path.join(image_NAR_root, filename)
            mask_dir = None
            name = filename[:-4]
            self.data_idx.append((image_dir, mask_dir, name))
            
        #random.shuffle(self.data_idx)
        
    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        image_dir, mask_dir, name = self.data_idx[idx]
        image = Image.open(image_dir)
        if mask_dir is not None:
            mask = Image.open(mask_dir)
            mask = np.array(mask)
        else:
            mask = np.zeros(image.size, dtype='uint8')
        
        
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask)  
        
        return image, mask, name


if __name__ == '__main__':
    dataset = IDRiD_sub1_dataset('./data/sub1/train', 'MA')
    print('dataset length: %d'%(len(dataset)))
    print('dataset sample')
    print(dataset[random.randint(0, len(dataset)-1)])
    