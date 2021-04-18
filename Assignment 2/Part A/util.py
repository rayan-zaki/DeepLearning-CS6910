import torch
import torch.nn as nn
from torchvision.transforms import RandomCrop, RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize, Compose
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import shutil

## Dataset info
iNaturalist = {
    'Normalize': {
        'mean': (0.485, 0.456, 0.406),
        'std':  (0.229, 0.224, 0.225)
    }
}

## Dataloaders
def data_loader(train_data, val_data, test_data, batchSize):
    train_dataLoader = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True)
    val_dataLoader = torch.utils.data.DataLoader(val_data, batch_size=batchSize, shuffle=True)
    test_dataLoader = torch.utils.data.DataLoader(test_data, batch_size=batchSize, shuffle=False)
    loaders = {
        'train' : train_dataLoader,
        'valid' : val_dataLoader,
        'test'  : test_dataLoader
    }
    return loaders


## transforms to match model input dims
def transform():
    
    resize = 224 #128#32
    val_resize = 256 #134 #36
    val_center_crop = resize
    
    #train_t = Compose([RandomResizedCrop(resize),
    train_t = Compose([RandomResizedCrop(resize),
                       RandomHorizontalFlip(),
                       ToTensor(),
                       Normalize(**iNaturalist['Normalize'])])
    valid_t = Compose([Resize(val_resize),
                       CenterCrop(resize),
                       ToTensor(),
                       Normalize(**iNaturalist['Normalize'])])
    test_t = Compose([Resize((resize,resize)), 
                      ToTensor(), 
                      Normalize(**iNaturalist['Normalize'])])
    
    transforms = {
        'training':   train_t,
        'validation': valid_t,
        'test': test_t
    }
    
    return transforms

## Load dataset fn
def load_datasets():
    transforms=transform()
    trainset  = torchvision.datasets.ImageFolder(r'C:\Users\Rayan Zaki\Downloads\inaturalist_12K\train_val\train', transforms['training'])
    valset    = torchvision.datasets.ImageFolder(r'C:\Users\Rayan Zaki\Downloads\inaturalist_12K\train_val\val', transforms['validation'])
    testset   = torchvision.datasets.ImageFolder(r'C:\Users\Rayan Zaki\Downloads\inaturalist_12K\val', transforms['test'])
    
    return trainset, valset, testset

## Save chkpt
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


## load chkpt
def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss from checkpoint to valid_loss
    valid_loss = checkpoint['valid_loss']
    # initialize valid_acc from checkpoint to valid_acc
    valid_acc = checkpoint['valid_acc']
    # initialize valid_loss from checkpoint to valid_loss
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss.item(), valid_acc, valid_loss_min


