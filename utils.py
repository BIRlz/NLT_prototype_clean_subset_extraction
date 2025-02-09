from functools import partial
from torchvision import transforms
from InceptionResNetV2 import *

from torch.utils.data import Dataset
from torchvision.models import resnet
from resnet_cifar import *

from PIL import Image

import math
import os
import numpy as np
import torch
import torch.nn as nn
from sinkhorn_distance import *
from PreResNet import *
import shutil

def save_code(save_file, save_path):
    shutil.copy(save_file, os.path.join(save_path, 'main.py'))

def save_meta_train_info(save_file, save_path, save_name):
    save_file = np.array(save_file)
    np.savetxt(os.path.join(save_path, save_name), save_file, fmt='%s')

def shuffle_batch(x, y):
    index = torch.randperm(x.size(0))
    x = x[index]
    y = y[index]
    return x, y, index

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    elr = args.e_lr
    fclr = args.fc_lr
    if args.cos:  # cosine lr schedule
        elr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        fclr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            elr *= 0.1 if epoch >= milestone else 1.
            fclr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['elr'] = elr
        param_group['fclr'] = fclr


def get_noisy_imbalanced_per_cls_list(nlt_list, cls_num):
    
    NLT_list = np.zeros(cls_num)

    for i in nlt_list:
        try:
            NLT_list[i] += 1
        except:
            import pdb;pdb.set_trace()
    
    return NLT_list


def get_idxs_per_cls_list(nlt_label, cls_num):
    
    NLT_list = [[] for _ in range(cls_num)]

    for i, nl in enumerate(nlt_label):
        NLT_list[nl].append(i)
    
    return NLT_list

def get_real_clean_per_cls_list(clean_labels, noisy_labels, cls_num):

    clean_idx = get_idxs_per_cls_list(clean_labels, cls_num)
    noisy_idx = get_idxs_per_cls_list(noisy_labels, cls_num)

    res = np.zeros(cls_num)

    for i, ni in enumerate(noisy_idx):
        for j_ni in ni:
            if j_ni in clean_idx[i]:
                res[i] += 1
    
    return res




class CustomWebvisionDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, img_paths, labels=None, transform=None, root=None):

        self.img_paths = img_paths
        self.train_label = labels
        self.transform = transform
        self.root = root

    def __getitem__(self, index):
        
        x = self.img_paths[index]
        x = os.path.join(self.root, x)

        x = Image.open(x).convert('RGB')   
        y = self.train_label[index]

        x = self.transform(x)

        return x, y, index

    def __len__(self):
        return self.img_paths.shape[0]

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, labels=None, transform=None, transform_batch=None):
        
        self.tensors = tensors.astype(np.uint8)
        self.transform = transform
        self._transform = transforms.ToPILImage()
        self.train_label = labels
        self.transform_batch = transform_batch

    def __getitem__(self, index):
        
        x = self.tensors[index]
        y = self.train_label[index]

        if len(x.shape) == 4:
            x = [self._transform(img) for img in x]
            x = torch.stack([self.transform(img) for img in x])

        else:
            x = self._transform(x)
            x = self.transform(x)

        return x, y, index

    def __len__(self):
        return self.tensors.shape[0]
    
train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
    ])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)
        
class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, class_num=10, feature_dim=128, arch=None, resume_ckpt=None, bn_splits=8):
        super(ModelBase, self).__init__()

        self.arch = arch

        if self.arch == 'ResNet18':
            # use split batchnorm
            norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
            resnet_arch = getattr(resnet, 'resnet18')
            net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)
            # net = resnet_arch(num_classes=feature_dim)

            self.net = []
            for name, module in net.named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if isinstance(module, nn.MaxPool2d):
                    continue
                if isinstance(module, nn.Linear):
                    self.net.append(nn.Flatten(1))
                self.net.append(module)

            self.net = nn.Sequential(*self.net)
            self.hidden_size = 512

            self.encoder = self.net
            self.classifier = nn.Linear(self.hidden_size, class_num)
            
            if resume_ckpt:
                self.encoder.load_state_dict(dict(resume_ckpt), strict=True)

            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

        else:
            self.net = resnet32(num_classes=class_num, use_norm=True)
            self.hidden_size = 64

            self.encoder = self.net
            self.classifier = nn.Linear(self.hidden_size, class_num)
            
            if resume_ckpt:
                self.encoder.load_state_dict(dict(resume_ckpt), strict=True)

            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            
        
    def forward(self, x):
        feat = self.encoder(x)
        if self.arch == 'ResNet32':
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(-1, self.hidden_size)
        out1 = self.classifier(feat)
        
        return feat, out1 


class WebVisionModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, class_num=10, feature_dim=128, arch=None, resume_ckpt=None, bn_splits=8):
        super(WebVisionModelBase, self).__init__()
        self.arch = arch

        if self.arch == 'InceptionResNetV2':
            self.net = InceptionResNetV2(num_classes=50)
            self.hidden_size = 1536
            self.classifier = nn.Linear(self.hidden_size, class_num)
            self.encoder = self.net

            if resume_ckpt:
                msg = self.encoder.load_state_dict(dict(resume_ckpt), strict=False)
                print(msg)

            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        if self.arch == 'resnet18':
            self.net = ResNet18(num_classes=class_num, low_dim=128, head='Linear')
            self.hidden_size = 512

            if resume_ckpt:
                msg = self.net.load_state_dict(dict(resume_ckpt), strict=False)
                print(msg)
            
            # self.classifier = self.net.linear2
            self.classifier = nn.Linear(self.hidden_size, class_num)
            self.encoder = nn.Sequential(*list(self.net.children()))[:-2]

        
    def forward(self, x):
        feat = self.encoder(x)
        
        if self.arch == 'resnet18':
            feat = F.adaptive_avg_pool2d(feat, (1, 1))

        feat = feat.view(-1, self.hidden_size)
        
        out1 = self.classifier(feat)
        
        return feat, out1 


class RedImangenetModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, class_num=10, arch=None, resume_ckpt=None):
        super(RedImangenetModelBase, self).__init__()
        self.arch = arch

        resnet_arch = getattr(resnet, 'resnet18')
        self.net = resnet_arch(num_classes=class_num)

        self.classifier = nn.Linear(512, class_num)
        self.encoder = nn.Sequential(*list(self.net.children()))[:-1]

        
    def forward(self, x):
        feat = self.encoder(x)
        out1 = self.classifier(feat)
        
        return feat, out1 


def pad_to_multiple_of_8(x, y):

    remainder = x.shape[0] % 8
    if remainder != 0:
        padding_size = 8 - remainder
    else:
        padding_size = 0

    if padding_size > 0:
        extra_padding_x = x[:padding_size]
        extra_padding_y = y[:padding_size]
    
        x = torch.cat([x, extra_padding_x], dim=0)
        y = torch.cat([y, extra_padding_y])


    return x, y


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0) 
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_weights(beta, img_num_list):
    
    #convert zero in img_num_list to one
    img_num_list = np.array(img_num_list, dtype=np.int64)
    img_num_list[img_num_list == 0] = 1
    
    effective_num = 1.0 - np.power(beta, img_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    return per_cls_weights