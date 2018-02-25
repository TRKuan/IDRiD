# -*- coding: utf-8 -*
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random


class IDRiD_sub1_dataset(Dataset):
    """
    Put the images into these directories respectivly:
        Apparent Retinopathy, No Apparent Retinopathy, MA, EX, HE, SE
            
    It will load in the original 4288x2848 image and crop it into 16x11 small images with 
    size of 256x256 on the fly.
    
    Use shuffle=False for this dataset as it caches only one image.
    
    Data Argumentation and transforms has not been implemented yet.
    """

    def __init__(self, root_dir):
        """
            Args:
                root_dir:the root directory of the images
        """
        self.task_type_list = ['MA', 'EX', 'HE', 'SE']
        self.root_dir = root_dir
        self.data_idx = []#(image_dir, mask_dirs, name)  mask_dirs is a list(None for NAR images)
        self.data_cache = {'image': None, 'mask': None, 'name': "", 'index': None}#cache the original size image

        image_root = os.path.join(self.root_dir, 'Apparent Retinopathy')
        image_NAR_root = os.path.join(self.root_dir, 'No Apparent Retinopathy')
        
        #Get the file index
        #AR images
        for filename in os.listdir(image_root):
            image_dir = os.path.join(image_root, filename)
            mask_dirs = {task_type:None for task_type in self.task_type_list}
            for task_type in self.task_type_list:
                m_dir = os.path.join(self.root_dir, task_type, filename[:-4]+'_'+task_type+'.tif')
                if os.path.isfile(m_dir): mask_dirs[task_type] = m_dir
            name = filename[:-4]
            self.data_idx.append((image_dir, mask_dirs, name))
        #NAR images
        for filename in os.listdir(image_NAR_root):
            image_dir = os.path.join(image_NAR_root, filename)
            mask_dirs = {task_type:None for task_type in self.task_type_list}
            name = filename[:-4]
            self.data_idx.append((image_dir, mask_dirs, name))
        
        #Shuffle
        random.shuffle(self.data_idx)
        
    def __len__(self):
        return len(self.data_idx)*16*11

    def __getitem__(self, idx):
        # crop the 4288x2848 image into 256x256 => 16x11 grid
        # 1 image => 16x11 = 176 small images
        n = int(idx/(11*16))#image index
        r = int((idx%(11*16))/16)#row
        c = (idx%(11*16))%16#column
        
        #Load the images if it's not in the cache
        if self.data_cache['index'] != n:
            image_dir, mask_dirs, name = self.data_idx[n]
            image = Image.open(image_dir)

            masks = []
            for task_type in self.task_type_list:
                if mask_dirs[task_type] is not None:
                    #AR images
                    mask = Image.open(mask_dirs[task_type])
                    mask = np.array(mask, dtype='float32')
                else:
                    #NAR images
                    w, h = image.size
                    mask = np.zeros((h, w), dtype='float32')
                masks.append(mask)
            masks = np.array(masks)
                
            self.data_cache = {'image': image, 'masks': masks, 'name': name, 'index': n}

        #crop the image
        image_crop = self.data_cache['image'].crop((c*256, r*256, c*256 + 256, r*256 + 256))
        masks_crop = self.data_cache['masks'][:, r*256:r*256+256, c*256:c*256+256]
        image_crop = transforms.ToTensor()(image_crop)
        masks_crop = torch.from_numpy(masks_crop)
        name = self.data_cache['name']+'(%d, %d)'%(r, c)
        
        return image_crop, masks_crop, name


if __name__ == '__main__':
    dataset = IDRiD_sub1_dataset('./data/sub1/train')
    print('dataset length: %d'%(len(dataset)))
    #data formate test
    print('dataset sample')
    image, mask, name = dataset[random.randint(0, len(dataset)-1)]
    print(image, mask, name)
    #show image test(need to comment out to tensor)
    '''
    import matplotlib.pyplot as plt
    for i in range(len(dataset)):
        image, mask, name = dataset[i]
        print(image, mask, name)
        plt.imshow(image)
        plt.show()
    '''
    
    # dataloader test
    '''
    from torch.utils.data import DataLoader
    import time
    t = time.time()
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
    for data in dataloader:
        pass
    print('%ds'%(time.time()-t))
    '''
    