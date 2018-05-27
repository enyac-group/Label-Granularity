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
import dataset


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--resume_dir', default=None, help='resume dir')
parser.add_argument('--superclass', default=None, help='one of the super class')
parser.add_argument('--gpus', default='0', help='gpus used')
parser.add_argument('--f2c', type=int, default=None, help='whether use coarse label')
parser.add_argument('--categories', default=None, help='which classes to use')
parser.add_argument('--data_ratio', type=float, default=1., help='ratio of training data to use')
parser.add_argument('--add_layer', type=int, default=0, help='whether to add additional layer')
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
    setup_logging(os.path.join(save_path, 'log_test.txt'))
logging.info("saving to %s", save_path)
logging.info("run arguments: %s", args)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



# Data
print('==> Preparing data..')

if args.categories == 'dog_cat':
    classes = (151, 153, 154, 157, 161, 281, 282, 283, 284, 285)
    classes_f2c = {}
    for idx,a_class in enumerate(classes):
        if a_class in [151, 153, 154, 157, 161]: # dogs
            classes_f2c[idx] = 0
        elif a_class in [281, 282, 283, 284, 285]: # cats
            classes_f2c[idx] = 1
elif args.categories == 'fruit_vege':
    classes = (949, 950, 951, 952, 953, 954, 955, 956, 936, 937, 938, 939, 942,
                943, 944, 945, 947)
    classes_f2c = {}
    for idx,a_class in enumerate(classes):
        if a_class in [949, 950, 951, 952, 953, 954, 955, 956]: # fruits
            classes_f2c[idx] = 0
        elif a_class in [936, 937, 938, 939, 942, 943, 944, 945, 947]: # vegetables
            classes_f2c[idx] = 1

if args.f2c == 1:
    NUM_CLASS = 2
elif args.f2c == 0:
    NUM_CLASS = len(classes_f2c)
else:
    raise ValueError

if args.f2c == 1 and args.add_layer == 1:
    fine_cls = len(classes_f2c)
else:
    fine_cls = None


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainset = dataset.data_imagenet.ImageFolder(root=None, train=True, class_list=classes, transform=transform_train, data_ratio=args.data_ratio)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = dataset.data_imagenet.ImageFolder(root=None, train=False, class_list=classes, transform=transform_test, data_ratio=1.)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.resume_dir)
    checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.t7'))
    net = checkpoint['net']
    if args.add_layer == 0:
        net.fine_cls = None
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG8')
    # net = ResNet18()
    net = PreActResNet18(num_classes=NUM_CLASS, thickness=16, blocks=[2,2,2,2], dropout=0., fine_cls=fine_cls)
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

# net = PreActResNet18(num_classes=2, thickness=64, blocks=[2,2,2,2])
# if args.resume:
#     assert os.path.isdir(args.resume_dir)
#     checkpoint = torch.load(os.path.join(args.resume_dir, 'ckpt.t7'))
#     net_dict = net.state_dict()
#     net_dict.update(checkpoint['net'].state_dict())
#     net_dict = {key: val for key,val in net_dict.items() if 'linear' not in key}
#     net.load_state_dict(net_dict, strict=False)


logging.info("model structure: %s", net)
num_parameters = sum([l.nelement() for l in net.parameters()])
logging.info("number of parameters: %d", num_parameters)

if use_cuda:
    net.cuda()
    logging.info('gpus: {}'.format([int(ele) for ele in args.gpus]))
    net = torch.nn.DataParallel(net, device_ids=[int(ele) for ele in args.gpus])
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# regime = {
#     0: {'optimizer': 'SGD', 'lr': 1e-2,
#         'weight_decay': 1e-4, 'momentum': 0.9},
#     60: {'lr': 1e-3},
#     90: {'lr': 1e-4},
#     120: {'lr': 1e-5},
# }
regime = {
    0: {'optimizer': 'SGD', 'lr': 1e-2,
        'weight_decay': 1e-4, 'momentum': 0.9},
    int(90//args.data_ratio): {'lr': 1e-3},
    int(135//args.data_ratio): {'lr': 1e-4},
    int(180//args.data_ratio): {'lr': 1e-5},
    int(210//args.data_ratio): {'lr': 1e-6},
}

logging.info('training regime: %s', regime)

# Training
def train(epoch, f2c=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    correct_f2c = 0
    total = 0
    global optimizer
    optimizer = adjust_optimizer(optimizer, epoch, regime)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if f2c:
            for idx,target in enumerate(targets):
                targets[idx] = classes_f2c[target]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if f2c == False:
            predicted_np = predicted.cpu().numpy()
            for idx,a_predicted in enumerate(predicted_np):
                predicted_np[idx] = classes_f2c[a_predicted]
            targets_np = targets.data.cpu().numpy()
            for idx,a_target in enumerate(targets_np):
                targets_np[idx] = classes_f2c[a_target]
            correct_f2c += (predicted_np == targets_np).sum()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 10 == 0:
            if f2c:
                logging.info('\n Epoch: [{0}][{1}/{2}]\t'
                            'Training Loss {train_loss:.3f} \t'
                            'Training Prec@1 {train_prec1:.3f} \t'
                            .format(epoch, batch_idx, len(trainloader),
                            train_loss=train_loss/(batch_idx+1), 
                            train_prec1=100.*correct/total))
            else:
                logging.info('\n Epoch: [{0}][{1}/{2}]\t'
                            'Training Loss {train_loss:.3f} \t'
                            'Training Prec@1 {train_prec1:.3f} \t'
                            'Training Prec@1 f2c {train_prec1_f2c:.3f} \t'
                            .format(epoch, batch_idx, len(trainloader),
                            train_loss=train_loss/(batch_idx+1), 
                            train_prec1=100.*correct/total,
                            train_prec1_f2c=100.*correct_f2c/total))


def test(epoch, f2c=False, train_f=True, testloader=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if f2c:
            for idx,target in enumerate(targets):
                targets[idx] = classes_f2c[target]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        if train_f and f2c:
            predicted_np = predicted.cpu().numpy()
            #print('predicted: {}'.format(predicted))
            #print('predicted_np: {}'.format(predicted_np))
            for idx,a_predicted in enumerate(predicted_np):
                predicted_np[idx] = classes_f2c[a_predicted]
            #correct += (predicted_np == targets.cpu().numpy()).sum()
            #print('targets: {}'.format(targets))
            correct += (predicted_np == targets.data.cpu().numpy()).sum()
        else:
            correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    logging.info('\n Epoch: {0}\t'
                    'Testing Loss {test_loss:.3f} \t'
                    'Testing Prec@1 {test_prec1:.3f} \t\n'
                    .format(epoch, 
                    test_loss=test_loss/len(testloader), 
                    test_prec1=100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and args.resume_dir is None:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(save_path, 'ckpt.t7'))
        best_acc = acc


#start_epoch = 0

# if args.f2c == 1:
#     for epoch in range(start_epoch, int(225//args.data_ratio)):
#         train(epoch, f2c=True)
#         test(epoch, f2c=True, train_f=False)
# elif args.f2c == 0:
#     for epoch in range(start_epoch, int(225//args.data_ratio)):
#         train(epoch, f2c=False)
#         test(epoch, f2c=False)
#         test(epoch, f2c=True, train_f=True)
    
if args.f2c == 1:
    logging.info('Test trainset: ')
    trainset_test = dataset.data_imagenet.ImageFolder(root=None, train=True, class_list=classes, transform=transform_test, data_ratio=args.data_ratio)
    trainloader_test = torch.utils.data.DataLoader(trainset_test, batch_size=100, shuffle=False, num_workers=2)
    test(start_epoch, f2c=True, train_f=False, loader=trainloader_test)

    logging.info('Test testset: ')
    test(start_epoch, f2c=True, train_f=False, loader=testloader)

elif args.f2c == 0:
    logging.info('Test trainset: ')
    trainset_test = dataset.data_imagenet.ImageFolder(root=None, train=True, class_list=classes, transform=transform_test, data_ratio=args.data_ratio)
    trainloader_test = torch.utils.data.DataLoader(trainset_test, batch_size=100, shuffle=False, num_workers=2)
    test(start_epoch, f2c=True, train_f=True, loader=trainloader_test)

    logging.info('Test testset: ')
    test(start_epoch, f2c=True, train_f=True, loader=testloader)
