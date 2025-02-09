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
parser.add_argument('--fc_lr', default=0.02, type=float, help='initial learning rate for fc', dest='fc_lr')
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

ot_criterion = SinkhornDistance_uniform(eps=0.1, max_iter=200, reduction='mean').to('cuda')
# ot_criterion_q = SinkhornDistance(eps=0.1, max_iter=200, reduction='mean').to('cuda')

if args.results_dir == '':
    args.results_dir = './debug/Webvision-FT-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

def OT_consistency_2(feats, labels, prototypes, prior_weights=None):

    if prior_weights is None:
        nu_prior = None
    else:
        nu_prior = prior_weights.softmax(-1)
    
    _ot_loss3, _pi3 = ot_criterion(feats, prototypes, nu=nu_prior)
    p_labels3 = torch.argmax(_pi3.log().softmax(-1), -1)

    return _ot_loss3, p_labels3, p_labels3 == labels, _pi3.log().softmax(-1)
        
def obtain_noised_features_prototype(net, data_loader, epoch, args):
    net.encoder.eval()
    net.classifier.eval()

    eval_bar = tqdm(data_loader)
    
    train_size = data_loader.dataset.train_label.shape[0]
    
    total_features = torch.zeros((train_size, args.hidden_size))
    total_labels = torch.zeros(train_size).long()

    unique_labels = torch.linspace(0, args.num_classes-1, args.num_classes).long()
    
    group_features = {}

    for ul in unique_labels:
        group_features.update({str(ul.cpu().numpy()): []})
    
    with torch.no_grad():
        for imgs, labels, index in eval_bar:
            
            imgs, labels = imgs.cuda(), labels.cuda() 

            feats, _ = model(imgs)
            
            for b, (lab, fs) in enumerate(zip(labels, feats)):
                group_features[str(lab.cpu().numpy())].append(fs.cpu().numpy())

                total_features[index[b]] = feats[b]
                total_labels[index[b]] = labels[b]

            eval_bar.set_description('Eval Epoch: [{}/{}]'.format(epoch, args.epochs))
    
    avg_features = torch.zeros((unique_labels.size(0), args.hidden_size))
    
    for keys, values in group_features.items():
        if not values.__len__() == 0:
            avg_features[int(keys)] = torch.from_numpy(np.array(values)).squeeze().mean(0)
    
    avg_features = avg_features.cuda()

    return avg_features

def train_2(net, data_loader, train_optimizer, criterion, scheduler, epoch, avg_prototypes, args):

    loss, train_bar = 0.0, tqdm(data_loader)
    total = 0.0
    correct = 0.0
    total_2 = 0.0

    total_p_true = 0
    total_p_equals_given = 0

    true_y = []
    true_x = []
    true_y_prob = []

    for imgs, labels, index in train_bar:
        train_optimizer.zero_grad()

        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
 
        net.encoder.eval()
        net.classifier.eval()
        
        feats, _ = net(imgs)

        if true_y.__len__() == 0:
            _temp_weights = None

        consistency_loss, ot_pseudo_labels, consistency_index, p_prob = OT_consistency_2(feats, labels, avg_prototypes, prior_weights=_temp_weights)

        true_index = torch.nonzero(consistency_index).squeeze()
        true_label = ot_pseudo_labels[true_index]
        true_x_index = index[true_index]
        true_label_prob = p_prob[true_index]

        if not len(true_index.shape) == 0:
            true_x.append(true_x_index.cpu().detach())
            true_y.append(true_label.cpu().detach())
            true_y_prob.append(true_label_prob.cpu().detach())

        if len(true_index.shape) == 0 or true_index.shape[0] <= 3:
            continue

        net.encoder.train()
        net.classifier.train()
        
        inputs, _labels = imgs[true_index], true_label

        _, outputs = net(inputs)

        l1 = criterion(outputs, _labels).sum() / labels.size(0)
        l2 = l1

        loss = l1 
            
        loss.backward()
        train_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(_labels.data).cpu().sum()
        
        total_p_true += torch.nonzero(ot_pseudo_labels == _clean_labels[index]).size(0)

        total_2 += (true_label == _clean_labels[true_x_index]).size(0)
        total_p_equals_given += torch.nonzero(true_label == _clean_labels[true_x_index]).size(0)

        train_bar.set_description('Estimating Epoch: [{}/{}], Train Acc: {:.6f}, ot_true: {:.6f}, ot_eq_given: {:.6f}, Loss-1: {:.4f}, Loss-2: {:.4f}'.format(
                                epoch, args.epochs, 100.*correct/total, 100.*total_p_true/total, 100.*total_p_equals_given/total_2, l1.item(), l2.item()))
    
    scheduler.step()
    
    true_x = torch.concat(true_x).cpu().numpy()
    true_y = torch.concat(true_y).cpu().numpy()
    true_y_prob = torch.concat(true_y_prob).cpu().numpy()
    
    return loss.item(), [true_x, true_y, true_y_prob]


# train for one epoch

def train_fc(net, data_loader, train_optimizer, criterion, epoch, args):
    net.encoder.train()
    net.classifier.train()

    loss, train_bar = 0.0, tqdm(data_loader)
    total = 0.0
    correct = 0.0

    for imgs, labels, index in train_bar:
        train_optimizer.zero_grad()

        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        if args.use_mixup and not args.use_cutmix:
            inputs, targets_a, targets_b, lam = mixup_data(imgs, labels, 0.5, use_cuda=True)

            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            feats, outputs = net(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
        
        # if not args.use_mixup and args.use_cutmix:
        lam = np.random.beta(0.2, 0.2)
        rand_index = torch.randperm(imgs.size()[0]).cuda()

        targets_a = labels
        targets_b = labels[rand_index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
        # # compute output
        feats, outputs = net(imgs)
                
        loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1. - lam)
        # loss = criterion(outputs, targets_a) 

        loss = loss.mean()

        loss.backward()
        train_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()    
        
        train_bar.set_description('Train FC: [{}/{}], Train Acc: {:.6f}, Loss-1: {:.4f}'.format(epoch, args.epochs, 100.*correct/total, loss.item()))   
    

def test(net, data_loader, criterion, epoch, args):
    net.eval()
    correct = 0
    total = 0
    total_preds = []
    total_targets = []
    total_test_loss = []
    

    loss, test_bar = 0.0, tqdm(data_loader)

    with torch.no_grad():
        for imgs, labels in test_bar:

            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            feats, outputs = net(imgs)
            
            _, predicted = torch.max(outputs, 1)            
            correct += predicted.eq(labels).cpu().sum().item()                 

            total_preds.append(predicted)
            total_targets.append(labels)

            test_loss = criterion(outputs, labels)
            total_test_loss.append(test_loss)
            
            total += imgs.size(0)

            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, correct / total * 100))

    acc = 100.*correct/total

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    cls_acc = [ round( 100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(args.num_classes)]
    
    return acc, cls_acc


def test_ot(net, data_loader, criterion, epoch, args, avg_prototypes):
    net.eval()
    correct = 0
    total = 0
    total_preds = []
    total_targets = []
    total_test_loss = []
    
    loss, test_bar = 0.0, tqdm(data_loader)

    with torch.no_grad():
        for imgs, labels in test_bar:

            imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            feats, outputs = net(imgs)
            
            consistency_loss, ot_pseudo_labels, consistency_index, p_prob = OT_consistency_2(feats, labels, avg_prototypes, prior_weights=None)

            # _, predicted = torch.max(outputs, 1)            
            correct += ot_pseudo_labels.eq(labels).cpu().sum().item()                 

            total_preds.append(ot_pseudo_labels)
            total_targets.append(labels)

            test_loss = criterion(outputs, labels)
            total_test_loss.append(test_loss)
            
            total += imgs.size(0)

            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, correct / total * 100))

    acc = 100.*correct/total

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    cls_acc = [ round( 100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(args.num_classes)]
    
    return acc, cls_acc


def update_eval_loader(train_data, transform, args, index_x, labels, drop_last=False, bs=512):

    
    eval_train_dataset = CustomWebvisionDataset(train_data[index_x], labels, transform=transform, root="/PATH/data/webvision/")

    eval_loader = DataLoader(
            eval_train_dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=drop_last,
            num_workers=12,
        )

    return eval_loader

def compute_weights(img_num_list):
    
    beta = 0.9

    #convert zero in img_num_list to one
    img_num_list = np.array(img_num_list, dtype=np.int64)
    img_num_list[img_num_list == 0] = 1
    
    effective_num = 1.0 - np.power(beta, img_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    return per_cls_weights

args.num_classes = 50
loader = dataloader.webvision_dataloader(batch_size=256, num_workers=12,root_dir="/PATH/data/webvision", log="log.txt", num_class=50)

train_data = []
train_labels = []

for path, label in loader.run('warmup').dataset.train_labels.items():
    train_data.append(path)
    train_labels.append(label)

train_labels = torch.tensor(train_labels).long()
train_data = np.array(train_data, dtype=object)
_clean_labels = copy.deepcopy(train_labels).cuda()


train_dataset = CustomWebvisionDataset(train_data, train_labels, transform=loader.transform_train, root="/PATH/data/webvision")

train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=12,
    )

pcl = get_noisy_imbalanced_per_cls_list(train_labels, args.num_classes)
print("Images Per Cls: ", pcl)

test_loader = loader.run('test')

encoder = []

if args.arch == 'InceptionResNetV2':
    args.resume = "/PATH/RoLT+/checkpoint/inception_webvision_best_s1.pth.tar"
    saved_ckpt = torch.load(args.resume, 'cuda')
    encoder = saved_ckpt['state_dict']

if args.arch == 'resnet18':
    args.resume = "/PATH/RoLT+/checkpoint/Sel-CL+_model.pth"
    saved_ckpt = torch.load(args.resume, 'cuda')
    args.hidden_size = 512
    for key in list(saved_ckpt.keys()):
        if key.startswith('module.'):
            _key = key.replace('module.', '')
            encoder.append((_key, saved_ckpt[key]))
    
    classifier_weight = saved_ckpt['module.linear2.weight']
    classifier_bias = saved_ckpt['module.linear2.bias']

model = WebVisionModelBase(class_num=args.num_classes, arch=args.arch, resume_ckpt=encoder)
# model.classifier.weight.data = classifier_weight
# model.classifier.bias.data = classifier_bias

model = model.cuda()

print(args)

param_list = [
    {'params': model.encoder.parameters(), 'lr': args.e_lr, 'momentum': 0.9, 'weight_decay': args.wd},
    {'params': model.classifier.parameters(), 'lr': args.fc_lr, 'momentum': 0.9, 'weight_decay': args.wd},
]

optimizer = torch.optim.SGD(param_list)
criterion = nn.CrossEntropyLoss(reduction='none')

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001, last_epoch=-1)

best_acc = 0.0
best_per_cls = None
epoch_start = 1
# logging
results = {'train_loss': [], 'test_acc@1': []}
if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
# dump args
with open(args.results_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)

index = np.arange(len(train_data))

clean_x = []
clean_y = []
clean_y_prob = []
p_r_a = []
sample_per_cls = []
acc_per_cls = []

eval_loader = update_eval_loader(train_data, loader.transform_test, args, index, train_labels[index], bs=256)
init_avg_prototypes = obtain_noised_features_prototype(model, eval_loader, 0, args)

avg_prototypes = init_avg_prototypes

save_code(__file__, args.results_dir)

for epoch in range(epoch_start, args.epochs + 1):

    train_loss, real_sample_index = train_2(model, train_loader, optimizer, criterion, scheduler, epoch, avg_prototypes, args)
    results['train_loss'].append(train_loss) 

    clean_x = torch.tensor(real_sample_index[0]).long().numpy()
    clean_y = train_labels[clean_x]
    clean_y_prob = real_sample_index[2]
    un_index = np.setdiff1d(index, clean_x)

    eval_loader = update_eval_loader(train_data, loader.transform_test, args, clean_x, clean_y, True, args.batch_size)
    unlabeled_trainloader = update_eval_loader(train_data, loader.transform_test, args, un_index, train_labels[un_index], True, args.batch_size)

    print("Clean Samples: {}".format(clean_x.shape[0]))
    print("Sample Per Class: {}".format(get_noisy_imbalanced_per_cls_list(clean_y, args.num_classes)))


    criterion_fc = nn.CrossEntropyLoss(reduction='none')    
    train_fc(model, eval_loader, optimizer, criterion_fc, epoch, args)


    # tmp_avg_prototypes = obtain_noised_features_prototype(model, eval_loader, 0, args)
    # lam = 0.99
    # avg_prototypes = lam * init_avg_prototypes + (1 - lam) * tmp_avg_prototypes
    # avg_prototypes = tmp_avg_prototypes

    #evaluate the model    
    test_acc_1, test_acc_per_cls = test(model, test_loader, criterion, epoch, args)
    print(test_acc_per_cls)
    
    results['test_acc@1'].append(test_acc_1)
    if test_acc_1 > best_acc:
        best_acc = test_acc_1
        best_per_cls = test_acc_per_cls
        
        best_ckpt = args.results_dir + '/model_best.pth'
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, best_ckpt)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
    # save model
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')
    print()

save_meta_train_info(p_r_a, args.results_dir, 'precision_recall_acc.txt')
save_meta_train_info(acc_per_cls, args.results_dir, 'acc_per_cls.txt')

print('Best Overall Acc: {}'.format(best_acc))
print('Best Per-Cls Acc: {}'.format(best_per_cls))
print("WebVision Test")

test_acc_top1, test_acc_top5, test_acc_per_cls = test(model, test_loader, criterion, epoch, args)
print(test_acc_top1)
print(test_acc_top5)
print(test_acc_per_cls)
    
imagenet_test_loader = loader.run('imagenet')
print("ImageNet Test")
test_acc_top1, test_acc_top5, test_acc_per_cls = test(model, imagenet_test_loader, criterion, epoch, args)
print(test_acc_top1)
print(test_acc_top5)
print(test_acc_per_cls)
    