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
import numpy as np
import seaborn as sns


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


# Plot confusion matrix
if args.resume:
    conf_matrix = pickle.load(open(os.path.join(args.resume_dir, 'conf_matrix.pkl'), 'rb'))

conf_matrix_nrm = conf_matrix / conf_matrix.sum(axis=0)
conf_matrix_nrm = (conf_matrix_nrm + np.transpose(conf_matrix_nrm)) / 2.
print('normalized confusion matrix: \n{}'.format(conf_matrix_nrm))
ax = sns.heatmap(conf_matrix_nrm, vmin=0, vmax=0.1)
fig = ax.get_figure()
fig.savefig(os.path.join(save_path, 'conf_matrix.png'))