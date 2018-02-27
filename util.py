# -*- coding: utf-8 -*
import torch
from sklearn.metrics import f1_score
import numpy as np
import os

def weighted_BCELoss(output, target, weights=None):

    output = output.clamp(min=1e-5, max=1-1e-5)
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.mean(loss)

def evaluate(y_true, y_pred):
    '''
    Calculate statistic matrix.
    
    Args:
        y_true:the pytorch tensor of ground truth
        y_pred:the pytorch tensor of prediction
    return:
        The F1 score
    '''
    y_true = y_true.numpy().flatten()
    y_pred = np.rint(y_pred.numpy().flatten())
    f1 = f1_score(y_true, y_pred)
    return f1


def save_model(model, save_dir, name):
    #save model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, name)
    print('Saving model to directory "%s"'%(path))
    torch.save(model.state_dict(), path)


