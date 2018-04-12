# -*- coding: utf-8 -*
from dataset import IDRiD_sub1_dataset
from model import GCN
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import os

use_gpu = torch.cuda.is_available
save_dir = "./saved_models"
model_name = "gcn_v5.pth"
data_dir = './data/sub1/val'
batch_size = 4

def show_image_sample():
    # dataset
    dataset = IDRiD_sub1_dataset(data_dir)

    
    #model
    model = GCN(4, 512)
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name))) 
    model.train(False)
    for n in range(12):
        #test
        full_image = np.zeros((3, 2848, 4288), dtype='float32')
        full_mask = np.zeros((4, 2848, 4288), dtype='float32')
        full_output = np.zeros((4, 2848, 4288), dtype='float32')#(C, H, W)
        title = ''
        for idx in range(9*6*n, 9*6*(n+1)):
            image, mask, name = dataset[idx]
            n = int(idx/(6*9))#image index
            r = int((idx%(6*9))/9)#row
            c = (idx%(6*9))%9#column
            title = name[:-8]
            
            if use_gpu:
                image = image.cuda()
                mask = mask.cuda()
            image, mask = Variable(image, volatile=True), Variable(mask, volatile=True)
                
            #forward
            output = model(image.unsqueeze(0))
            output = F.sigmoid(output)
            output = output[0]
            if c < 8:
                if r == 5:
                    full_output[:, r*512:r*512+512-224, c*512:c*512+512] = output.cpu().data.numpy()[:, :-224, :]
                    full_mask[:, r*512:r*512+512-224, c*512:c*512+512] = mask.cpu().data.numpy()[:, :-224, :]
                    full_image[:, r*512:r*512+512-224, c*512:c*512+512] = image.cpu().data.numpy()[:, :-224, :]

                else:
                    full_output[:, r*512:r*512+512, c*512:c*512+512] = output.cpu().data.numpy()
                    full_mask[:, r*512:r*512+512, c*512:c*512+512] = mask.cpu().data.numpy()
                    full_image[:, r*512:r*512+512, c*512:c*512+512] = image.cpu().data.numpy()

            
        full_image = full_image.transpose(1, 2, 0)
        MA = full_output[0]
        EX = full_output[1]
        HE = full_output[2]
        SE = full_output[3]
        
                
        plt.figure()
        plt.axis('off')
        plt.suptitle(title)
        plt.subplot(331)
        plt.title('image')
        fig = plt.imshow(full_image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(332)
        plt.title('ground truth MA')
        fig = plt.imshow(full_mask[0])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(333)
        plt.title('ground truth EX')
        fig = plt.imshow(full_mask[1])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(334)
        plt.title('ground truth HE')
        fig = plt.imshow(full_mask[2])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(335)
        plt.title('ground truth SE')
        fig = plt.imshow(full_mask[3])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(336)
        plt.title('predict MA')
        fig = plt.imshow(MA)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(337)
        plt.title('predict EX')
        fig = plt.imshow(EX)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(338)
        plt.title('predict HE')
        fig = plt.imshow(HE)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(339)
        plt.title('predict SE')
        fig = plt.imshow(SE)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        
        
        plt.show()

class save_predict_dataset(Dataset):


    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_idx = []#(image_dir, mask_dirs, name)  mask_dirs is a list(None for NAR images)
        self.data_cache = {'image': None, 'name': "", 'index': None}#cache the original size image

        
        #Get the file index
        for filename in os.listdir(root_dir):
            image_dir = os.path.join(root_dir, filename)
            name = filename[:-4]
            self.data_idx.append((image_dir, name))
        
    def __len__(self):
        return len(self.data_idx)*6*9

    def __getitem__(self, idx):
        # crop the 4288x2848 image into 512x512 => 9x6 grid
        # 1 image => 9x6 = 54 small images
        n = int(idx/(6*9))#image index
        r = int((idx%(6*9))/9)#row
        c = (idx%(6*9))%9#column
        
        #Load the images if it's not in the cache
        if self.data_cache['index'] != n:
            image_dir, name = self.data_idx[n]
            image = Image.open(image_dir)
                
            self.data_cache = {'image': image, 'name': name, 'index': n}
            
        
        #crop the image
        
        image_crop = self.data_cache['image'].crop((c*512, r*512, c*512 + 512, r*512 + 512))
        image_crop = transforms.ToTensor()(image_crop)
        name = self.data_cache['name']
        
        return image_crop, name

def save_output(root_dir, output_dir):
    # dataset
    dataset = save_predict_dataset(root_dir)

    
    #model
    model = GCN(4, 512)
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name))) 
    model.train(False)
    for n in range(int(len(dataset)/(6*9))):
        #test
        full_output = np.zeros((4, 2848, 4288), dtype='float32')#(C, H, W)
        title = ''
        for idx in range(6*9*n, 6*9*(n+1)):
            image, name = dataset[idx]
            r = int((idx%(6*9))/9)#row
            c = (idx%(6*9))%9#column
            title = name
            
            if use_gpu:
                image = image.cuda()
            image = Variable(image, volatile=True)
                
            #forward
            output = model(image.unsqueeze(0))
            output = F.sigmoid(output)
            output = output[0]
            
            if c < 8:
                if r == 5:
                    full_output[:, r*512:r*512+512-224, c*512:c*512+512] = output.cpu().data.numpy()[:, :-224, :]
                else:
                    full_output[:, r*512:r*512+512, c*512:c*512+512] = output.cpu().data.numpy()                    
        
        for i, d in enumerate(['MA', 'EX', 'HE', 'SE']):
            if not os.path.exists(os.path.join(output_dir, d)):
                os.makedirs(os.path.join(output_dir, d))
            im = np.expand_dims(full_output[i], axis=0).transpose(1, 2, 0)
            im = full_output[i]*255
            im = np.uint8(im)
            im = Image.fromarray(im)
            im.save(os.path.join(output_dir, d, title+'.jpg'))
        

def run_statistic(threshold):
    '''
        evaluate on small images result
    '''
    # dataset
    dataset = IDRiD_sub1_dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #print('Data: %d'%(len(dataset)))
    
    #model
    model = GCN(4, 512)
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name))) 
    model.train(False)
    for i in range(4):
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
            #for i in range(len(outputs)):
            y_pred = outputs[i]
            y_true = masks[i]
            y_pred = y_pred.numpy().flatten()
            y_pred = np.where(y_pred > threshold, 1, 0)
            y_true = y_true.numpy().flatten()
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
            
            #verbose
            if idx%5==0 and idx!=0:
                print('\r{:.2f}%'.format(100*idx/len(dataloader)), end='\r')
        #print()
        type_list = ['MA', 'EX', 'HE', 'SE']
        precision, recall, f1, _ = precision_recall_fscore_support(np.array(y_true_list).flatten(), np.array(y_pred_list).flatten(), average='binary')
        print('{}    \nThreshold: {:.2f}\nPrecision: {:.4f}\nRecall: {:.4f}\nF1: {:.4f}'.format(type_list[i], threshold, precision, recall, f1))

def evaluate(threshold):
    '''
        evaluate results with original image size
    '''
    task_type_list = ['MA', 'EX', 'HE', 'SE']
    result = []
    print('-------------------')
    for i in range(4):
        print('--------')
        mean = [0, 0, 0, 0]
        
        for filename in os.listdir('./data/sub1/val/Apparent Retinopathy'):
            gt_dirs = {task_type:None for task_type in task_type_list}
            for task_type in task_type_list:
                m_dir = os.path.join('./data/sub1/val', task_type, filename[:-4]+'_'+task_type+'.tif')
                if os.path.isfile(m_dir): gt_dirs[task_type] = m_dir
            pd_dirs = {task_type:None for task_type in task_type_list}
            for task_type in task_type_list:
                m_dir = os.path.join('./data/sub1/predict', task_type, filename[:-4]+'.jpg')
                if os.path.isfile(m_dir): pd_dirs[task_type] = m_dir
            gts = []
            for task_type in task_type_list:
                mask = Image.open(gt_dirs[task_type])
                mask = np.array(mask, dtype='float32')
                gts.append(mask)
            gts = np.array(gts[i])
            pds = []
            for task_type in task_type_list:
                mask = Image.open(pd_dirs[task_type])
                mask = np.array(mask, dtype='float32')
                mask /= 255
                pds.append(mask)
            pds = np.array(pds[i])
            pds = np.where(pds > threshold, 1, 0)
            tn, fp, fn, tp = confusion_matrix(gts.flatten(), pds.flatten()).ravel()
            ppv = tp/(tp+fp)
            sensitivity = tp/(tp+fn)
            specificity = tn/(tn+fp)
            f1 = (2*tp)/(2*tp+fp+fn)
                        
            mean[0] += ppv
            mean[1] += sensitivity
            mean[2] += specificity
            mean[3] += f1
        
        for filename in os.listdir('./data/sub1/val/No Apparent Retinopathy'):
            gt_dirs = {task_type:None for task_type in task_type_list}
            for task_type in task_type_list:
                m_dir = os.path.join('./data/sub1/val', task_type, filename[:-4]+'_'+task_type+'.tif')
                if os.path.isfile(m_dir): gt_dirs[task_type] = m_dir
            pd_dirs = {task_type:None for task_type in task_type_list}
            for task_type in task_type_list:
                m_dir = os.path.join('./data/sub1/predict', task_type, filename[:-4]+'.jpg')
                if os.path.isfile(m_dir): pd_dirs[task_type] = m_dir
            
            gts = []
            for task_type in task_type_list:
                mask = mask = np.zeros((2848, 4288), dtype='float32')
                gts.append(mask)
            gts = np.array(gts[i])
            
            pds = []
            for task_type in task_type_list:
                mask = Image.open(pd_dirs[task_type])
                mask = np.array(mask, dtype='float32')
                mask /= 255
                pds.append(mask)
            pds = np.array(pds[i])
            pds = np.where(pds > threshold, 1, 0)
            
            
            try: 
                tn, fp, fn, tp = confusion_matrix(gts.flatten(), pds.flatten()).ravel()
                ppv = 0
                sensitivity = 0
                specificity = tn/(tn+fp)
                f1 = 0
            except: 
                ppv = 0
                sensitivity = 0
                specificity = 0
                f1 = 0
            
            mean[0] += ppv
            mean[1] += sensitivity
            mean[2] += specificity
            mean[3] += f1
        
        print(task_type_list[i])
        print('Threshold: {:.2f}\nPPV: {:.4f}\nSensitivity: {:.4f}\nSpecificity: {:.4f}\nF1: {:.4f}'.format(threshold, mean[0]/6, mean[1]/6, mean[2]/12, mean[3]/6))
        result.append((mean[0]/6, mean[1]/6, mean[2]/12, mean[3]/6))
        
    avg = [0, 0, 0, 0]
    for r in result:
        for i in range(4):
            avg[i]+=r[i]
        
    print("---------")
    print('Average')
    print('Threshold: {:.2f}\nPPV: {:.4f}\nSensitivity: {:.4f}\nSpecificity: {:.4f}\nF1: {:.4f}'.format(threshold, avg[0]/4, avg[1]/4, avg[2]/4, avg[3]/4))
        
if __name__ == '__main__':
    #save_output('./data/sub1/val/Apparent Retinopathy', '../data/sub1/predict')
    #print(model_name)
    #run_statistic(0.3)
    show_image_sample()
    '''
    for th in [0.3]:
        evaluate(th)
    '''
    
