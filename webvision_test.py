from datetime import datetime
from torch.utils.data import DataLoader
from resnet_cifar import *
from tqdm import tqdm
import argparse
import json
import ot
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from sinkhorn_distance import *
from torch.autograd import Variable
import dataloader_webvision as dataloader
import torch.cuda.amp as amp 

import copy
# import torchnet


parser = argparse.ArgumentParser()

# parser.add_argument('-a', '--arch', default='InceptionResNetV2')
parser.add_argument('-a', '--arch', default='resnet18')

parser.add_argument('--dataset', default='webvision50', help='dataset')
parser.add_argument('--hidden_size', default=1536, help='hidden_size')

parser.add_argument('--e_lr', default=0.001, type=float, help='initial learning rate for encoder', dest='e_lr')
parser.add_argument('--fc_lr', default=0.01, type=float, help='initial learning rate for fc', dest='fc_lr')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--use_mixup', default=False, type=bool, help='use mixup')
parser.add_argument('--use_cutmix', default=False, type=bool, help='use cutmix')
parser.add_argument('--data_split', default='imb', type=str, help='data split type')

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, help='path to cache (default: none)')

args = parser.parse_args()  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 150
args.cos = False
args.schedule = []  # cos in use
args.symmetric = False

def test(ff, gg, data_loader, args):
    ff.eval()
    gg.eval()
    net.eval()
    model.eval()
    total_preds = []
    total_targets = []
    
    loss, test_bar = 0.0, tqdm(data_loader)

    with torch.no_grad():
        for imgs, labels in test_bar:

            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            if args.arch == 'InceptionResNetV2':
                feats = ff(imgs).squeeze()
                outputs = gg(feats)
            else:
                
                feats = ff(imgs)
                feats = F.adaptive_avg_pool2d(feats, (1,1)).squeeze()
                outputs = gg(feats)

            total_preds.append(outputs)
            total_targets.append(labels)

            test_bar.set_description('Testing...')

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)

    accuracy_top = accuracy(total_preds, total_targets, (1, 5))

    cls_acc = [ round( 100. * ((total_preds.max(-1)[1] == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(args.num_classes)]
    

    return accuracy_top[0], accuracy_top[1], cls_acc


args.num_classes = 50
loader = dataloader.webvision_dataloader(batch_size=32, num_workers=12, root_dir="/PATH/data/webvision", log="log.txt", num_class=50)

test_loader = loader.run('test')
imagenet_test_loader = loader.run('imagenet')

encoder = []
classifier = []

if args.arch == 'InceptionResNetV2':
    # args.resume = "/PATH/RoLT+/debug/Webvision-FT-2024-01-09-13-29-37/model_best.pth"
    args.resume = "/PATH/RoLT+/debug/Webvision-FT-2024-01-09-14-39-55/model_best.pth"
    saved_ckpt = torch.load(args.resume, 'cuda')
    ckpt = saved_ckpt['state_dict']

    for key in list(ckpt.keys()):
        if key.startswith('encoder.'):
            _key = key.replace('encoder.', '')
            encoder.append((_key, ckpt[key]))
        if key.startswith('classifier'):
            _key = key.replace('classifier.', '')
            classifier.append((_key, ckpt[key]))

    net = InceptionResNetV2(num_classes=50)
    ff = nn.Sequential(*list(net.children())[:-1])
    gg = nn.Linear(1536, 50)

    ff.load_state_dict(dict(encoder), True)
    gg.load_state_dict(dict(classifier), True)
    model = nn.Sequential(ff, gg)

if args.arch == 'resnet18':
    args.resume = "/PATH/RoLT+/debug/Webvision-FT-2024-01-24-06-41-32/model_best.pth"
    saved_ckpt = torch.load(args.resume, 'cuda')
    ckpt = saved_ckpt['state_dict']
    args.hidden_size = 512

    for key in list(ckpt.keys()):
        if key.startswith('encoder.'):
            _key = key.replace('encoder.', '')
            encoder.append((_key, ckpt[key]))

        if key.startswith('classifier'):
            _key = key.replace('classifier.', '')
            classifier.append((_key, ckpt[key]))

    net = ResNet18(num_classes=50, low_dim=128, head='Linear').cuda()

    ff = nn.Sequential(*list(net.children())[:-2])
    gg = nn.Linear(512, 50)

    ff.load_state_dict(dict(encoder), True)
    gg.load_state_dict(dict(classifier), True)
    model = nn.Sequential(ff, gg)


model = model.cuda()

for params in model.parameters():
    params.requires_grad = False

criterion = nn.CrossEntropyLoss(reduction='none')

print("WebVision Test")
test_acc_top1, test_acc_top5, test_acc_per_cls = test(ff, gg, test_loader, args)
print(test_acc_top1)
print(test_acc_top5)
print(test_acc_per_cls)
    
print("ImageNet Test")
test_acc_top1, test_acc_top5, test_acc_per_cls = test(ff, gg, imagenet_test_loader, args)
print(test_acc_top1)
print(test_acc_top5)
print(test_acc_per_cls)
    