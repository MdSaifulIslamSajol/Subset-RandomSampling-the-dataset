# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:01:10 2022

@author: msajol1
"""

import torch
from torchvision import transforms
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets
import numpy as np
from pathlib import Path
from torch.utils.data import RandomSampler, DataLoader, Subset
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
#%%
batchsize=10
# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
transform =transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                       std=[0.229, 0.224, 0.225]),
])
#%%
train_set_cifar = datasets.CIFAR10(root='CIFAR10', transform=transform, download=True, train=True)
test_set_cifar = datasets.CIFAR10(root='CIFAR10', transform=transform, download=True, train=False)

#%% variables and transform
batchsize=100
epochs = 30 # Number of epochs
num_train_samples= 5000  #319837
num_test_samples= 200  #39996

# check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%%  defining dataset path

print("train_set_cifar length :", len(train_set_cifar))
print("test_set_cifar length :", len(test_set_cifar))

# =============================================================================
# #%% Using Random Sampling data
# =============================================================================
# torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)

#train
random_sampled_train_set= torch.utils.data.RandomSampler(train_set_cifar, 
                                                                replacement=False, 
                                                                num_samples=num_train_samples, 
                                                                generator=None)

random_sampled_train_set_docu_dataloader  = torch.utils.data.DataLoader(
                                            dataset=train_set_cifar, 
                                             sampler=random_sampled_train_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

# test
random_sampled_test_set = torch.utils.data.RandomSampler(test_set_cifar, 
                                                                replacement=False, 
                                                                num_samples=num_test_samples, 
                                                                generator=None)


random_sampled_test_set_docu_dataloader  = torch.utils.data.DataLoader(
                                            dataset=test_set_cifar, 
                                             sampler=random_sampled_test_set,
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

# Validation
train_targets = []
for images, target in random_sampled_train_set_docu_dataloader:
    train_targets.append(target)
train_targets = torch.cat(train_targets)

print("\n Class labels and their corresponding counts: \n")
print(train_targets.unique(return_counts=True))
print("len(random_sampled_train_set):",len(random_sampled_train_set))
print("len(random_sampled_test_set):",len(random_sampled_test_set))
#%% spliting using random split
train_dataset1, train_dataset2 = random_split(train_set_cifar, (5000, 45000))
print("len(train_dataset1:)",len(train_dataset1))
print("len(train_dataset2):",len(train_dataset2))

# =============================================================================
# #%%  Using Subset  : divide the dataset by a factor 5
# =============================================================================

train_set_cifar_half = torch.utils.data.Subset(train_set_cifar, range(0, len(train_set_cifar), 5)) # divide the dataset by 5
print("len(train_set_cifar_half:)",len(train_set_cifar_half))


train_loader  = torch.utils.data.DataLoader(
                                            dataset=train_set_cifar_half, 
                                           
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

# Validation
train_targets = []
for images, target in train_loader:
    train_targets.append(target)
train_targets = torch.cat(train_targets)

print("\n Class labels and their corresponding counts: \n")
print(train_targets.unique(return_counts=True))

# =============================================================================
# #%% filtering by classes  
# =============================================================================

# select classes you want to include in your subset
classes = torch.tensor([0, 1, 2, 3, 4])

# get indices that correspond to one of the selected classes
indices = (torch.tensor(train_set_cifar.targets)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]

# subset the dataset
train_subset_cifar = torch.utils.data.Subset(train_set_cifar, indices)

train_loader  = torch.utils.data.DataLoader(
                                            dataset=train_subset_cifar,                                           
                                            batch_size=batchsize,
                                            shuffle=False,
                                            num_workers=0)

# Validation
train_targets = []
for images, target in train_loader:
    train_targets.append(target)
train_targets = torch.cat(train_targets)

print("\n Class labels and their corresponding counts: \n")
print(train_targets.unique(return_counts=True))

#%%

