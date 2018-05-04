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
import sklearn.metrics.pairwise
import numpy as np
import pickle


parser = argparse.ArgumentParser(description='Gen feat from net pre-trained on ImageNet')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--resume_dir', default=None, help='resume dir')
parser.add_argument('--gpus', default='0', help='gpus used')
args = parser.parse_args()

args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if args.resume_dir is None:
    print('resume_dir is None')
    save_path = os.path.join(args.results_dir, args.save)
else:
    print('resume_dir is not None')
    save_path = args.resume_dir
if not os.path.exists(save_path):
    os.makedirs(save_path)
if args.resume_dir is None:
    setup_logging(os.path.join(save_path, 'log.txt'))
else:
    setup_logging(os.path.join(save_path, 'log_eval.txt'))
logging.info("saving to %s", save_path)
logging.info("run arguments: %s", args)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])


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
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18(num_classes=10, thickness=16)
    # net = Auto_encoder()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    net = resnet50_imagenet(pretrained=True)

logging.info("model structure: %s", net)
num_parameters = sum([l.nelement() for l in net.parameters()])
logging.info("number of parameters: %d", num_parameters)

if use_cuda:
    net.cuda()
    logging.info('gpus: {}'.format([int(ele) for ele in args.gpus]))
    net = torch.nn.DataParallel(net, device_ids=[int(ele) for ele in args.gpus])
    cudnn.benchmark = True

def get_feat(loader):
    net.eval()
    all_feats = []
    all_targets = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        all_targets.append(targets.numpy())
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        _, feats = net(inputs)
        all_feats.append(feats.data.cpu().numpy())
    all_feats = np.vstack(all_feats)
    all_targets = np.hstack(all_targets)
    return all_feats, all_targets

trainset_unshuffle = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=True, download=True, transform=transform_test)
trainloader_unshuffle = torch.utils.data.DataLoader(trainset_unshuffle, batch_size=250, shuffle=False, num_workers=2)

feats, targets = get_feat(trainloader_unshuffle)
pickle.dump({'feats': feats, 'targets': targets}, 
            open(os.path.join(save_path, 'resnet50_feats.pkl'), 'wb'))
