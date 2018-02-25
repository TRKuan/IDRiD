# -*- coding: utf-8 -*
from dataset import IDRiD_sub1_dataset
from util import evaluate, save_model
from model import FCN8s
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import copy

use_gpu = torch.cuda.is_available
save_dir = "./saved_models"
model_name = "test.pth"
data_train_dir = './data/sub1/train'
data_val_dir = './data/sub1/val'
batch_size = 4
num_epochs = 10
lr = 1e-4

def make_dataloaders(batch_size=batch_size):
    dataset_train = IDRiD_sub1_dataset(data_train_dir)
    dataset_val = IDRiD_sub1_dataset(data_val_dir)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    print('Training data: %d\nValidation data: %d'%((len(dataset_train)), len(dataset_val)))
    return dataloaders

def train_model(model, num_epochs, dataloaders, criterion, optimizer):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
        
            running_loss = 0.0
            running_acc = 0.0
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
                
                outputs = F.sigmoid(outputs).cpu().data#remenber to apply sigmoid befor usage
                masks = masks.cpu().data
                for i in range(len(outputs)):
                    y_pred = outputs[i]
                    y_true = masks[i]
                    running_acc += evaluate(y_pred, y_true)
                
                #verbose
                if idx%5==0 and idx!=0:
                    print('\r{} {:.2f}%'.format(phase, 100*idx/len(dataloaders[phase])), end='')
     
            print()
            
            epoch_loss = running_loss / data_num
            epoch_acc = running_acc / data_num
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # dataset
    dataloaders = make_dataloaders(batch_size=batch_size)
    
    #model
    model = FCN8s(4)
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    
    #training
    criterion = nn.BCEWithLogitsLoss(size_average=True)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    model = train_model(model, num_epochs, dataloaders, criterion, optimizer)
    
    #save
    save_model(model, save_dir, model_name)
    
    