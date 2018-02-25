# -*- coding: utf-8 -*
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import os

def evaluate(y_pred, y_true):
    '''
    Calculate statistic matrix.
    
    Args:
        y_pred:the pytorch tensor of prediction
        y_true:the pytorch tensor of ground truth
    '''
    y_pred = np.rint(y_pred.numpy().flatten())
    y_true = y_true.numpy().flatten()
    acc = accuracy_score(y_true, y_pred)
    return acc
    '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    if tp+fn == 0: sensitivity = 0.0
    else: sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    ppv = tp/(tp+fp)
    return sensitivity, specificity, ppv, accuracy
    '''

def save_model(model, save_dir, name):
    #save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, name)
    print('Saving model to directory "%s"'%(path))
    torch.save(model.state_dict(), path)


