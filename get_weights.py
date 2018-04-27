from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import dataset

import os
import argparse

from models import *
from utils import progress_bar, adjust_optimizer, setup_logging
from torch.autograd import Variable
from datetime import datetime
import logging
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Getting params')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--resume_dir', default=None, help='resume dir')
args = parser.parse_args()

#args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#save_path = os.path.join(args.results_dir, args.save)
#if not os.path.exists(save_path):
#    os.makedirs(save_path)
save_path = args.results_dir

checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.t7'))

#checkpoint = torch.load(os.path.join('results/2018-04-26_00-15-11', 'ckpt.t7'))

params = list(checkpoint['net'].parameters())
print(params[-2].size(), params[-1].size())
pickle.dump([params[-2], params[-1]], open(os.path.join(save_path, 'last_layer_params.pkl'), 'wb'))
