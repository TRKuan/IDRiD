# -*- coding: utf-8 -*
from dataset import IDRiD_sub1_dataset
from model import FCN8s
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import os

use_gpu = torch.cuda.is_available
save_dir = "./saved_models"
model_name = "fcn8s.pth"
data_dir = './data/sub1/val'
batch_size = 4

def show_image_sample():
    # dataset
    dataset = IDRiD_sub1_dataset(data_dir)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    
    #model
    model = FCN8s(4)
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name))) 
    model.train(False)
    
    #test
    full_image = np.zeros((3, 2848, 4288), dtype='float32')
    full_mask = np.zeros((4, 2848, 4288), dtype='float32')
    full_output = np.zeros((4, 2848, 4288), dtype='float32')#(C, H, W)
    title = ''
    for idx in range(11*16):
        image, mask, name = dataset[idx]
        n = int(idx/(11*16))#image index
        r = int((idx%(11*16))/16)#row
        c = (idx%(11*16))%16#column
        title = name[:-8]
        
        if use_gpu:
            image = image.cuda()
            mask = mask.cuda()
        image, mask = Variable(image, volatile=True), Variable(mask, volatile=True)
            
        #forward
        output = model(image.unsqueeze(0))
        output = F.sigmoid(output)
        output = output[0]
        
        full_image[:, r*256:r*256+256, c*256:c*256+256] = image.cpu().data.numpy()
        full_mask[:, r*256:r*256+256, c*256:c*256+256] = mask.cpu().data.numpy()
        full_output[:, r*256:r*256+256, c*256:c*256+256] = output.cpu().data.numpy()
        
    full_image = full_image.transpose(1, 2, 0)
    MA = full_output[0]
    EX = full_output[1]
    HE = full_output[2]
    SE = full_output[3]
    
    plt.figure()
    plt.suptitle(title)
    plt.subplot(331)
    plt.title('image')
    plt.imshow(full_image)
    plt.subplot(332)
    plt.title('ground truth MA')
    plt.imshow(full_mask[0])
    plt.subplot(333)
    plt.title('ground truth EX')
    plt.imshow(full_mask[1])
    plt.subplot(334)
    plt.title('ground truth HE')
    plt.imshow(full_mask[2])
    plt.subplot(335)
    plt.title('ground truth SE')
    plt.imshow(full_mask[3])
    plt.subplot(336)
    plt.title('predict MA')
    plt.imshow(MA)
    plt.subplot(337)
    plt.title('predict EX')
    plt.imshow(EX)
    plt.subplot(338)
    plt.title('predict HE')
    plt.imshow(HE)
    plt.subplot(339)
    plt.title('predict SE')
    plt.imshow(SE)

    plt.show()

def run_statistic():
    # dataset
    dataset = IDRiD_sub1_dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Data: %d'%(len(dataset)))
    
    #model
    model = FCN8s(4)
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name))) 
    model.train(False)
    
    y_pred_list = []
    y_true_list = []
    for idx, data in enumerate(dataloader):
        images, masks, names = data
        
        if use_gpu:
            images = images.cuda()
            masks = masks.cuda()
        images, masks = Variable(images, volatile=True), Variable(masks, volatile=True)
        
        #forward
        outputs = model(images)
        
        # statistics                
        outputs = F.sigmoid(outputs).cpu().data#remenber to apply sigmoid befor usage
        masks = masks.cpu().data
        for i in range(len(outputs)):
            y_pred = outputs[i]
            y_true = masks[i]
            y_pred = np.rint(y_pred.numpy().flatten())
            y_true = y_true.numpy().flatten()
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
        
        #verbose
        if idx%5==0 and idx!=0:
            print('\r{:.2f}%'.format(100*idx/len(dataloader)), end='')
    print()
    
    recall = recall_score(np.array(y_pred_list).flatten(), np.array(y_true_list).flatten())
    print('Recall: {:.4f}'.format(recall))

    
    

if __name__ == '__main__':
    run_statistic()
    show_image_sample()

    