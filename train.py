# -*- coding: utf-8 -*
from dataset import IDRiD_sub1_dataset
from util import CrossEntropyLoss2d, save_model
from model import Model
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import time

use_gpu = torch.cuda.is_available
save_dir = "./saved_models"
data_train_dir = './data/sub1/train'
data_val_dir = './data/sub1/val'
task_type = 'MA'
batch_size = 2
num_epochs = 10

def make_dataloaders(batch_size=batch_size):
    dataset_train = IDRiD_sub1_dataset(data_train_dir, task_type)
    #dataset_val = IDRiD_sub1_dataset(data_val_dir, task_type)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    #dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4)
    #dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    #print('Training data: %d\nValidation data: %d'%(len(dataset_train)), len(dataset_val))
    dataloaders = {'train': dataloader_train}
    print('Training data: %d'%(len(dataset_train)))
    return dataloaders

def train_model(model, num_epochs, dataloaders, criterion, optimizer):
    since = time.time()
    best_model_wts = model.state_dict()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        #for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
        
        running_loss = 0.0
        data_num = 0
        
        for idx, data in enumerate(dataloaders[phase]):
            images, masks, names = data

            if use_gpu:
                images = images.cuda()
                masks = masks.cuda()
            if phase == 'train':
                images, masks = Variable(images, volatile=False), Variable(masks, volatile=False)
            else:
                images, masks = Variable(images, volatile=True), Variable(masks, volatile=True)

            optimizer.zero_grad()
            
            #forward
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            #backword
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            # statistics
            
            running_loss += loss.data[0] * images.size(0)
            data_num += images.size(0)
            
        
        epoch_loss = running_loss / data_num
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        '''
        # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        '''
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    dataloaders = make_dataloaders(batch_size=batch_size)
    
    model = Model()
    if use_gpu:
        model = model.cuda()
    
    criterion = CrossEntropyLoss2d()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    model = train_model(model, num_epochs, dataloaders, criterion, optimizer)
    save_model(model, save_dir, "net.pth")
    
    