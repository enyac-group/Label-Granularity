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


parser = argparse.ArgumentParser(description='Auto-encoder on CIFAR-10')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--resume_dir', default=None, help='resume dir')
parser.add_argument('--superclass', default=None, help='one of the super class')
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
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)

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
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18(num_classes=10, thickness=16)
    net = Auto_encoder()
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
    logging.info('gpus: {}'.format([int(ele) for ele in args.gpus]))
    net = torch.nn.DataParallel(net, device_ids=[int(ele) for ele in args.gpus])
    cudnn.benchmark = True

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# regime = {
#     0: {'optimizer': 'SGD', 'lr': 1e-1,
#         'weight_decay': 5e-4, 'momentum': 0.9},
#     150: {'lr': 1e-2},
#     250: {'lr': 1e-3}
# }
regime = {
    0: {'optimizer': 'SGD', 'lr': 1e-1,
        'weight_decay': 5e-4, 'momentum': 0.9},
    100: {'lr': 1e-2},
    150: {'lr': 1e-3}
}
logging.info('training regime: %s', regime)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    global optimizer
    optimizer = adjust_optimizer(optimizer, epoch, regime)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, _ = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        if batch_idx % 10 == 0:
            logging.info('\n Epoch: [{0}][{1}/{2}]\t'
                        'Training Loss {train_loss:.3f} \t'
                        .format(epoch, batch_idx, len(trainloader),
                        train_loss=train_loss/(batch_idx+1)))

# Training
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

def test_accuracy_autoencoder(train_feat, test_feat, train_label, test_label):
	Dist = sklearn.metrics.pairwise.pairwise_distances(train_feat, test_feat)
	nearest_n = np.argmin(Dist, axis=0)
	pred_label = train_label[nearest_n]
	return np.sum(pred_label == test_label)/float(test_feat.shape[0])

trainset_unshuffle = torchvision.datasets.CIFAR10(root='/home/rzding/DATA', train=True, download=True, transform=transform_test)
trainloader_unshuffle = torch.utils.data.DataLoader(trainset_unshuffle, batch_size=250, shuffle=False, num_workers=2)
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    feats, targets = get_feat(trainloader_unshuffle)
    feats_test, targets_test = get_feat(testloader)
    pickle.dump(feats, open(os.path.join(save_path, 'ae_feats.pkl')))
    acc = test_accuracy_autoencoder(feats, feats_test, targets, targets_test)
    logging.info('\n Epoch: [{}]\t Accuracy: {} \t'
                        .format(epoch, acc))




feats, targets = get_feat(trainloader_unshuffle)
pickle.dump(feats, open(os.path.join(save_path, 'ae_feats.pkl')))
