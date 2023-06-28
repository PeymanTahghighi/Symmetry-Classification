#===============================================================
#===============================================================
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits, cross_entropy
import config
#===============================================================
#===============================================================

#===============================================================
def dice_loss(logits, 
                true, 
                eps=1e-7, 
                sigmoid = False,
                multilabel = False,
                arange_logits = False):

    if sigmoid is True:
        logits = torch.sigmoid(logits);
    
    if arange_logits is True:
        logits = logits.permute(0,2,3,1);

    dims = (1,2,3);

    intersection = torch.sum(true * logits, dims);
    union = torch.sum(true + logits, dims);
    d_loss = torch.mean((2.0*intersection) / (union + eps));
    return 1-d_loss;
#===============================================================

#===============================================================
def focal_loss(logits,
                true,
                alpha = 0.8,
                gamma = 2.0,
                arange_logits = False,
                mutual_exclusion = False):

    if mutual_exclusion is False:
        if arange_logits is True:
            logits = logits.permute(0,2,3,1);
        
        bce = binary_cross_entropy_with_logits(logits.squeeze(dim=3), true.float(), reduction='none');
        bce_exp = torch.exp(-bce);
        f_loss = torch.mean(alpha * (1-bce_exp)**gamma*bce);
        return f_loss;
    else:
        logits = logits.permute(0,2,3,1).reshape(-1,3);
        true = true.view(-1);
        ce_loss = cross_entropy(logits, true.long(), reduction='none');
        p = torch.softmax(logits, axis = 1);
        #true_one_hot = one_hot(true.long(), Config.NUM_CLASSES);
        p = torch.take_along_dim(p, true.long().unsqueeze(dim = 1), dim = 1).squeeze();
        #p = torch.index_select(p, dim = 3, index = true);
        #assuming true is a one hot vector
        #ce_loss = -torch.log(p * true_one_hot + 1e-6);
        focal_mul = (1-p)**gamma;
        f_loss = focal_mul * ce_loss;
        return torch.mean(f_loss);
#===============================================================

#===============================================================
def tversky_loss(logits,
                true,
                alpha = 2.0,
                beta = 1.0,
                sigmoid = False,
                arange_logits = False,
                smooth = 1,
                mutual_exclusion = False):
    
    if arange_logits is True:
        
        logits = logits.permute(0,2,3,1);
    
    if mutual_exclusion is False:
        if sigmoid is True:
            logits = torch.sigmoid(logits);
    else:
        true = one_hot(true.long(), 3);
        logits = torch.softmax(logits,dim=3);
        true = true.squeeze(dim = 3);
        
    if true.dim() < 4:
        true = true.unsqueeze(dim=3)
    
    dims = (1,2,3);
    tp = torch.sum(logits * true, dims);
    fp = torch.sum((1-logits) * true, dims);
    fn = torch.sum(logits * (1-true), dims);
    tversky = torch.mean((tp + smooth) / (tp + alpha*fp + beta*fn + smooth))  
    return 1-tversky;
#===============================================================