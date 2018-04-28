# To run this, pay attention to this:
# define num_classes when initializing the model
# define f2c when calling train() and test()

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, adjust_optimizer, setup_logging
from torch.autograd import Variable
from datetime import datetime
import logging


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#trainset = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=True, download=True, transform=transform_train)
trainset = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=True, download=True, transform=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

#testset = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=False, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=False, download=True, transform=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_f2c = {}
for idx,a_class in enumerate(classes):
    if a_class in ['bird', 'plane', 'car', 'ship', 'truck']:
        classes_f2c[idx] = 0
    elif a_class in ['cat', 'deer', 'dog', 'frog', 'horse']:
        classes_f2c[idx] = 1

for batch_idx, (inputs, targets) in enumerate(trainloader):
    if batch_idx == 0:
        print(inputs)
        print(targets)