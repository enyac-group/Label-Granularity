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
import pickle


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--resume_dir', default=None, help='resume dir')
parser.add_argument('--superclass', default=None, help='one of the super class')
parser.add_argument('--gpus', default='0', help='gpus used')
args = parser.parse_args()

args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = os.path.join(args.results_dir, args.save)
if not os.path.exists(save_path):
    os.makedirs(save_path)
setup_logging(os.path.join(save_path, 'log.txt'))
logging.info("saving to %s", save_path)
logging.info("run arguments: %s", args)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

trainset_unshuffle = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=True, download=True, transform=transform_test)
trainloader_unshuffle = torch.utils.data.DataLoader(trainset_unshuffle, batch_size=250, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_f2c = {}
for idx,a_class in enumerate(classes):
    if a_class in ['plane', 'car', 'ship', 'truck']:
        classes_f2c[idx] = 0
    elif a_class in ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']:
        classes_f2c[idx] = 1

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.resume_dir)
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.t7'))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    raise ValueError

logging.info("model structure: %s", net)
num_parameters = sum([l.nelement() for l in net.parameters()])
logging.info("number of parameters: %d", num_parameters)

if use_cuda:
    net.cuda()
    logging.info('gpus: {}'.format([int(ele) for ele in args.gpus]))
    net = torch.nn.DataParallel(net, device_ids=[int(ele) for ele in args.gpus])
    cudnn.benchmark = True


def conf_matrix(net, loader, num_classes=10):
    net.eval()
    all_outputs = []
    all_targets = []
    cls_as = np.zeros((num_classes, num_classes))
    for batch_idx, (inputs, targets) in enumerate(loader):
        targ_cls = targets.numpy()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        pred_cls = outputs.data.cpu().numpy().argmax(axis=1)
        assert len(pred_cls.shape) == 1
        for i in range(pred_cls.shape[0]):
            cls_as[targ_cls[i], pred_cls[i]] += 1

    return cls_as


matrix = conf_matrix(net, testloader)
print('confusion matrix: \n{}'.format(matrix))
pickle.dump(matrix, open(os.path.join(save_path, 'conf_matrix.pkl'), 'wb'))

