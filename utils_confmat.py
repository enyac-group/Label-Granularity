'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import dataset

def conf_matrix(net, loader, num_classes=10):
    net.eval()
    all_outputs = []
    all_targets = []
    cls_as = np.zeros((num_classes, num_classes))
    for batch_idx, (inputs, targets) in enumerate(loader):
        targ_cls = targets.numpy()
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs, _ = net(inputs)
        pred_cls = outputs.data.cpu().numpy().argmax(axis=1)
        assert len(pred_cls.shape) == 1
        # print('targ: {}'.format(targ_cls))
        # print('pred: {}'.format(pred_cls))
        for i in range(pred_cls.shape[0]):
            cls_as[targ_cls[i], pred_cls[i]] += 1

    return cls_as

# def inter_conf(conf_mat, group):
#     conf_list = []
#     for i in range(len(conf_mat)):
#         for j in range(i+1, len(conf_mat[i])):
#             if not ([i,j] in group or [j,i] in group):
#                 conf_list.append(conf_mat[i,j])
#     print('there are {} class pairs not in the same group'.format(len(conf_list)))
#     return sum(conf_list) / len(conf_list)

# def intra_conf(conf_mat, group):
#     conf_list = []
#     for i in range(len(conf_mat)):
#         for j in range(i+1, len(conf_mat[i])):
#             if ([i,j] in group or [j,i] in group):
#                 conf_list.append(conf_mat[i,j])
#     print('there are {} class pairs in the same group'.format(len(conf_list)))
#     return sum(conf_list) / len(conf_list)

# def confusion(net, loader, classes_f2c):
#     matrix = conf_matrix(net, loader, num_classes=len(classes_f2c))
#     conf_matrix_nrm = matrix / matrix.sum(axis=0)
#     conf_matrix_nrm = (conf_matrix_nrm + np.transpose(conf_matrix_nrm)) / 2.
#     print('normalized confusion matrix: \n{}'.format(conf_matrix_nrm))
#     num_coarse_classes = max([classes_f2c[f] for f in classes_f2c]) + 1
#     print('classes_f2c: {}'.format(classes_f2c))
#     print('num_coarse_classes: {}'.format(num_coarse_classes))
#     group = [[f for f in classes_f2c if classes_f2c[f] == c_cls] for c_cls in range(num_coarse_classes)]
#     print('group: {}'.format(group))
#     inter_confusion = inter_conf(conf_matrix_nrm, group=group)
#     intra_confusion = intra_conf(conf_matrix_nrm, group=group)
#     return inter_confusion, intra_confusion

def inter_conf(conf_mat, group):
    conf_list = []
    for i in range(len(conf_mat)):
        for j in range(i+1, len(conf_mat[i])):
            if group[i] != group[j]:
                conf_list.append(conf_mat[i,j])
    print('there are {} class pairs not in the same group'.format(len(conf_list)))
    return sum(conf_list) / len(conf_list)

def intra_conf(conf_mat, group):
    conf_list = []
    for i in range(len(conf_mat)):
        for j in range(i+1, len(conf_mat[i])):
            if group[i] == group[j]:
                conf_list.append(conf_mat[i,j])
    print('there are {} class pairs in the same group'.format(len(conf_list)))
    return sum(conf_list) / len(conf_list)

def confusion(net, loader, classes_f2c):
    matrix = conf_matrix(net, loader, num_classes=len(classes_f2c))
    conf_matrix_nrm = matrix / matrix.sum(axis=0)
    conf_matrix_nrm = (conf_matrix_nrm + np.transpose(conf_matrix_nrm)) / 2.
    print('normalized confusion matrix: \n{}'.format(conf_matrix_nrm))
    inter_confusion = inter_conf(conf_matrix_nrm, group=classes_f2c)
    intra_confusion = intra_conf(conf_matrix_nrm, group=classes_f2c)
    return inter_confusion, intra_confusion