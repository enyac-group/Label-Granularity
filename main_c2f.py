# To run this, pay attention to this:
# define num_classes when initializing the model
# define f2c when calling train() and test()
# define num_clusters

'''Train CIFAR10 with PyTorch.'''
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

NUM_CLASSES = 2
NUM_CLUSTERS = 5

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--resume_dir', default=None, help='resume dir')
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

trainset = dataset.data_cifar10_red.CIFAR10_RED(root='/home/rzding/DATA', train=True, download=True, transform=transform_train)
#trainset = dataset.data_cifar10.CIFAR10(root='/home/rzding/DATA', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

testset = dataset.data_cifar10_red.CIFAR10_RED(root='/home/rzding/DATA', train=False, download=True, transform=transform_test)
#testset = dataset.data_cifar10.CIFAR10(root='/home/rzding/DATA', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.resume_dir)
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.t7'))
    net = checkpoint['net']
    #best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = PreActResNet18(num_classes=10)
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

logging.info("model structure: %s", net)
num_parameters = sum([l.nelement() for l in net.parameters()])
logging.info("number of parameters: %d", num_parameters)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True



# Step1: start from a pre-trained model, load it and save the output of last layer
# result format: a matrix where each row is a datapoint, a vector as class of each datapoint
def get_feat(net, trainloader):
    net.eval()
    all_feats = []
    all_idx = []
    all_targets = []
    for batch_idx, (inputs, input_idx, targets) in enumerate(trainloader):
        all_idx.append(input_idx.numpy())
        all_targets.append(targets.numpy())
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, feats = net(inputs)
        all_feats.append(feats.data.cpu().numpy())
    all_feats = np.vstack(all_feats)
    all_idx = np.hstack(all_idx)
    all_targets = np.hstack(all_targets)
    return all_feats, all_idx, all_targets


trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2)
train_feats, train_idx, all_targets = get_feat(net, trainloader)
pickle.dump(train_feats, open(os.path.join(save_path, 'train_feats.pkl'), 'wb'))
print('all feats size: {}'.format(train_feats.shape))
print('feats sum: {}'.format(train_feats.sum(axis=1)))
print('feats first row: {}'.format(train_feats[0]))
#pickle.dump([train_idx, all_targets], open(os.path.join(save_path, 'debug.pkl'), 'wb'))
pickle.dump([None, all_targets], open(os.path.join(save_path, 'debug.pkl'), 'wb'))

# Step2: cluster the data points per class
import sklearn.cluster as cls

def clustering(train_data, num_clusters):
    cluster_algo = cls.SpectralClustering(n_clusters=num_clusters, n_jobs=-1)	
    cluster_algo.fit(train_data)
    return cluster_algo.labels_.reshape(-1)

def normalize_c(x):
    #return (x - x.mean(axis=0)) / x.std(axis=0)
    return x / np.linalg.norm(x, ord=2, axis=0, keepdims=True)

def normalize_r(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

label_f = np.zeros(len(all_targets))
for a_class in range(NUM_CLASSES):
    idx = (all_targets == a_class)
    #label_cur = clustering(train_feats[idx] / train_feats[idx].max(), num_clusters=NUM_CLUSTERS)
    label_cur = clustering(normalize_c(train_feats[idx]), num_clusters=NUM_CLUSTERS)
    label_cur = label_cur + NUM_CLUSTERS * a_class
    label_f[idx] = label_cur

#label_f = np.hstack(label_f)
#print('before sorting:', label_f)
label_f = label_f[train_idx.argsort()]
#print('after sorting:', label_f)
pickle.dump(label_f, open(os.path.join(save_path, 'label_f.pkl'), 'wb'))

# # Step3: use the new label to train network
# # Training
# net_new = PreActResNet18(num_classes=NUM_CLASSES*NUM_CLUSTERS)
# if use_cuda:
#     net_new.cuda()
#     net_new = torch.nn.DataParallel(net_new, device_ids=[0])
#     cudnn.benchmark = True

# # init from pre-trained model
# # net_new_dict = net_new.state_dict()
# # net_new_dict.update(net.state_dict())
# # net_new.load_state_dict(net_new_dict)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net_new.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# regime = {
#     0: {'optimizer': 'SGD', 'lr': 1e-1,
#         'weight_decay': 5e-4, 'momentum': 0.9},
#     150: {'lr': 1e-2},
#     250: {'lr': 1e-3},
#     350: {'lr': 1e-4}
# }
# logging.info('training regime: %s', regime)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
# def train(epoch, net_new, trainloader, optimizer, fine=False):
#     print('\nEpoch: %d' % epoch)
#     net_new.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     optimizer = adjust_optimizer(optimizer, epoch, regime)
#     for batch_idx, (inputs, input_idx, targets) in enumerate(trainloader):
#         if fine:
#             for idx,target in enumerate(targets):
#                 #print(targets[idx], int(label_f[input_idx[idx]]))
#                 targets[idx] = int(label_f[input_idx[idx]])
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         optimizer.zero_grad()
#         inputs, targets = Variable(inputs), Variable(targets)
#         outputs, feats = net_new(inputs)
#         #print(outputs.data.cpu().numpy())
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.data[0]
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         correct += predicted.eq(targets.data).cpu().sum()
#         #print('targets: ', targets.data)
#         #print('predicted: ', predicted)

#         #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#         #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
#         if batch_idx % 20 == 0:
#             logging.info('\n Epoch: [{0}][{1}/{2}]\t'
#                         'Training Loss {train_loss:.3f} \t'
#                         'Training Prec@1 {train_prec1:.3f} \t'
#                         .format(epoch, batch_idx, len(trainloader),
#                         train_loss=train_loss/(batch_idx+1), 
#                         train_prec1=100.*correct/total))


# def test(epoch, net_new, testloader, fine=False, train_f=True):
#     global best_acc
#     net_new.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, input_idx, targets) in enumerate(testloader):
#         if fine:
#             raise ValueError
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()
#         inputs, targets = Variable(inputs, volatile=True), Variable(targets)
#         outputs, feats = net_new(inputs)
#         loss = criterion(outputs, targets)

#         test_loss += loss.data[0]
#         _, predicted = torch.max(outputs.data, 1)
#         total += targets.size(0)
#         if train_f and not fine:
#             predicted_np = predicted.cpu().numpy()
#             #print('predicted: {}'.format(predicted))
#             #print('predicted_np: {}'.format(predicted_np))
#             for idx,a_predicted in enumerate(predicted_np):
#                 predicted_np[idx] = a_predicted // NUM_CLUSTERS
#             #correct += (predicted_np == targets.cpu().numpy()).sum()
#             #print('targets: {}'.format(targets))
#             correct += (predicted_np == targets.data.cpu().numpy()).sum()
#         else:
#             correct += predicted.eq(targets.data).cpu().sum()

#         #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#         #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
#     logging.info('\n Epoch: {0}\t'
#                     'Testing Loss {test_loss:.3f} \t'
#                     'Testing Prec@1 {test_prec1:.3f} \t\n'
#                     .format(epoch, 
#                     test_loss=test_loss/len(testloader), 
#                     test_prec1=100.*correct/total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net_new.module if use_cuda else net_new,
#             'acc': acc,
#             'epoch': epoch,
#         }
#         torch.save(state, os.path.join(save_path, 'ckpt.t7'))
#         best_acc = acc

# start_epoch = 0
# for epoch in range(start_epoch, start_epoch+300):
#     train(epoch, net_new, trainloader, optimizer, fine=True)    
#     test(epoch, net_new, testloader, fine=False, train_f=True)
