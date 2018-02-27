# -*- coding: utf-8 -*
from dataset import IDRiD_sub1_dataset
from util import evaluate, save_model, weighted_BCELoss
from model import GCN
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import copy

use_gpu = torch.cuda.is_available
save_dir = "./saved_models"
model_name = "gcn_v3.pth"
data_train_dir = './data/sub1/train'
data_val_dir = './data/sub1/val'
batch_size = 24
num_epochs = 150
lr = 1e-4

def make_dataloaders(batch_size=batch_size):
    dataset_train = IDRiD_sub1_dataset(data_train_dir)
    dataset_val = IDRiD_sub1_dataset(data_val_dir)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    print('Training data: %d\nValidation data: %d'%((len(dataset_train)), len(dataset_val)))
    return dataloaders

def train_model(model, num_epochs, dataloaders, optimizer, scheduler):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
        
            running_loss = 0.0
            running_f1 = 0.0
            data_num = 0
            
            for idx, data in enumerate(dataloaders[phase]):
                images, masks, names = data
                
                #weight for loss
                weights = [5, 1]
                if use_gpu:
                    weights = torch.FloatTensor(weights).cuda()

    
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
                outputs = F.sigmoid(outputs)#remenber to apply sigmoid befor usage
                loss = weighted_BCELoss(outputs, masks, weights)
                
                #backword
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # statistics
                running_loss += loss.data[0]*images.size(0)
                data_num += images.size(0)
                outputs = outputs.cpu().data
                masks = masks.cpu().data
                running_f1 += evaluate(masks, outputs)*images.size(0)
                
                #verbose
                if idx%5==0 and idx!=0:
                    print('\r{} {:.2f}%'.format(phase, 100*idx/len(dataloaders[phase])), end='\r')
     
            #print()
            epoch_loss = running_loss / data_num
            epoch_f1 = running_f1 / data_num
            if phase == 'val':
                scheduler.step(epoch_loss)
            print('{} Loss: {:.4f} F1 score: {:.4f}'.format(phase, epoch_loss, epoch_f1))
            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                save_model(model, save_dir, model_name)
        
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    print('Best F1 score: {:.4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # dataset
    dataloaders = make_dataloaders(batch_size=batch_size)
    
    #model
    model = GCN(4, 256)
    if use_gpu:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()
    
    #training
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    model = train_model(model, num_epochs, dataloaders, optimizer, scheduler)
    
    #save
    save_model(model, save_dir, model_name)
    
    