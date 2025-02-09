from tabasco_data_utils import *
from datetime import datetime
from torch.utils.data import DataLoader
from resnet_cifar import *
from tqdm import tqdm
import argparse
import json
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from sinkhorn_distance import *
from torch.autograd import Variable
import dataloader_cifar as dataloader
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score
import copy

parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

parser.add_argument('-a', '--arch', default='resnet18')

# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--e_lr', default=0.01, type=float, help='initial learning rate for encoder', dest='e_lr')
parser.add_argument('--fc_lr', default=1, type=float, help='initial learning rate for fc', dest='fc_lr')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--use_mixup', default=False, type=bool, help='use mixup')
parser.add_argument('--use_cutmix', default=False, type=bool, help='use cutmix')

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, help='path to cache (default: none)')

args = parser.parse_args('')  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 100
args.cos = False
args.schedule = []  # cos in use
args.symmetric = False

ot_criterion = SinkhornDistance(eps=0.1, max_iter=200, dis='cos', reduction='mean').to('cuda')

if args.results_dir == '':
    args.results_dir = './debug/Supervised-FT-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

def OT_consistency_2(feats, labels, prototypes):


    # compute OT loss
    bb = torch.ones(prototypes.size(0)).cuda() / prototypes.size(0)

    _ot_loss, _pi, C, U, V = ot_criterion(feats, prototypes, bb)

    p_labels2 = torch.argmax(_pi.log().softmax(-1), -1)
    
    return _ot_loss, p_labels2, p_labels2 == labels, _pi.log().softmax(-1)
        
def obtain_noised_features_prototype(net, data_loader, epoch, args):
    net.encoder.eval()
    net.classifier.eval()

    eval_bar = tqdm(data_loader)
    
    train_size = data_loader.dataset.train_label.shape[0]
    
    total_features = torch.zeros((train_size, 512))
    total_labels = torch.zeros(train_size).long()
    unique_labels = torch.unique(torch.tensor(data_loader.dataset.train_label))

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
    
    avg_features = torch.zeros((unique_labels.size(0), 512))
    
    for keys, values in group_features.items():
        avg_features[int(keys)] = torch.from_numpy(np.array(values)).squeeze().mean(0)
    
    avg_features = avg_features.cuda()

    return avg_features


# train for one epoch
def train(net, data_loader, train_optimizer, criterion, scheduler, epoch, avg_prototypes, args):
    net.encoder.train()
    net.classifier.train()

    loss, train_bar = 0.0, tqdm(data_loader)
    total = 0.0
    correct = 0.0
    total_2 = 0.0

    total_p_true = 0
    total_p_equals_given = 0

    true_y = []
    true_x = []

    for imgs, labels, index in train_bar:
        train_optimizer.zero_grad()

        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        feats, outputs = net(imgs)
        lam = 1
        targets_a = labels
        targets_b = labels

        consistency_loss, ot_pseudo_labels, consistency_index, p_prob = OT_consistency_2(feats, labels, avg_prototypes)

        true_index = torch.nonzero(consistency_index).squeeze()
        true_label = ot_pseudo_labels[true_index]
        true_x_index = index[true_index]

        l1 = (criterion(outputs, ot_pseudo_labels) * consistency_index.float()).mean()
                    
        l2 = (1 * F.binary_cross_entropy_with_logits(outputs, p_prob) * (1 - consistency_index.float())).mean()

        true_x.append(true_x_index.cpu())
        true_y.append(true_label.cpu())

        loss = l1 
            
        loss.backward()
        train_optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()
        
        total_p_true += torch.nonzero(ot_pseudo_labels == _clean_labels[index]).size(0)

        total_2 += (true_label == _clean_labels[true_x_index]).size(0)
        total_p_equals_given += torch.nonzero(true_label == _clean_labels[true_x_index]).size(0)

        train_bar.set_description('Train Epoch: [{}/{}], Train Acc: {:.6f}, ot_true: {:.6f}, ot_eq_given: {:.6f}, Loss-1: {:.4f}, Loss-2: {:.4f}'.format(epoch, args.epochs, 100.*correct/total, 100.*total_p_true/total, 100.*total_p_equals_given/total_2, l1.item(), l2.item()))
    
    scheduler.step()
    true_x = torch.concat(true_x).cpu().numpy()
    true_y = torch.concat(true_y).cpu().numpy()
    
    return loss.item(), [true_x, true_y]


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
                for i in range(10)]
    
    return acc, cls_acc

def update_eval_loader(train_data, transform, args, index_x, labels, drop_last=False):

    eval_train_dataset = CustomTensorDataset(train_data[index_x], labels, transform=transform)

    eval_loader = DataLoader(
            eval_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=12,
        )

    return eval_loader

def compute_weights(img_num_list):
    
    beta = 0.99

    #convert zero in img_num_list to one
    img_num_list = np.array(img_num_list, dtype=np.int64)
    img_num_list[img_num_list == 0] = 1
    
    effective_num = 1.0 - np.power(beta, img_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    return per_cls_weights

imb_factor = 0.01
noise_ratio = 0.5

loader = dataloader.cifar_dataloader('cifar10', imb_type='exp', imb_factor=imb_factor, noise_mode='imb', noise_ratio=noise_ratio,\
    batch_size=args.batch_size, num_workers=5, root_dir='./data/cifar-10-batches-py', log='log.txt')

train_data = torch.Tensor(loader.run('warmup').dataset.train_data).cpu().numpy()
clean_labels = torch.Tensor(loader.run('warmup').dataset.clean_targets).long()
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()

train_dataset = CustomTensorDataset(train_data, noisy_labels, transform=train_transform)

pcl = get_noisy_imbalanced_per_cls_list(noisy_labels, 10)
_clean_labels = copy.deepcopy(clean_labels).cuda()

print("Images Per Cls: ", pcl)

train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
    )

test_loader = loader.run('test')

if args.arch == "resnet18":
    args.resume = "/PATH/RoLT+/cache-2023-10-30-08-35-58-moco/model_best.pth"
else:
    #resnet32
    args.resume = "/PATH/RoLT+/cache-2023-10-30-05-07-30-moco/model_best.pth"

saved_ckpt = torch.load(args.resume, 'cuda')
encoder = []
for key in list(saved_ckpt['state_dict'].keys()):
    if key.startswith('encoder_q'):
        _key = key.replace('encoder_q.net.', '')
        encoder.append((_key, saved_ckpt['state_dict'][key]))

model = ModelBase(arch=args.arch, resume_ckpt=encoder)
model = model.cuda()

print(args)

param_list = [
    {'params': model.encoder.parameters(), 'lr': args.e_lr, 'momentum': 0.9, 'weight_decay': args.wd},
    {'params': model.classifier.parameters(), 'lr': args.fc_lr, 'momentum': 0.9, 'weight_decay': args.wd},
]

optimizer = torch.optim.SGD(param_list)
criterion = nn.CrossEntropyLoss(reduction='none')

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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

index = np.arange(train_data.shape[0])

clean_x = []
clean_y = []
init_avg_prototypes = None

eval_loader = update_eval_loader(train_data, test_transform, args, index, noisy_labels[index])
avg_prototypes = obtain_noised_features_prototype(model, eval_loader, 0, args)

save_code(__file__, args.results_dir)

for epoch in range(epoch_start, args.epochs + 1):

   #train the model
    train_loss, real_sample_index = train(model, train_loader, optimizer, criterion, scheduler, epoch, avg_prototypes, args)
    results['train_loss'].append(train_loss)

    # clean_x = torch.cat([torch.tensor(clean_x), torch.tensor(real_sample_index[0])], 0)
    # clean_y = torch.cat([torch.tensor(clean_y), torch.tensor(real_sample_index[1])], 0)

    # _, unique_indices = np.unique(clean_x.numpy(), return_index=True)

    # clean_x = clean_x[unique_indices].long().numpy()
    # clean_y = clean_y[unique_indices].long()

    clean_x = torch.tensor(real_sample_index[0]).long().numpy()
    clean_y = noisy_labels[clean_x]
    
    _precision = precision_score(clean_y.cpu(), _clean_labels[clean_x].cpu(), average='macro')
    _recall = recall_score(clean_y.cpu(), _clean_labels[clean_x].cpu(), average='macro')
    _acc = accuracy_score(clean_y.cpu(), _clean_labels[clean_x].cpu())

    print("Precision: {:.4f} Recall: {:.4f} Acc: {:.4f}".format(_precision, _recall, _acc))

    eval_loader = update_eval_loader(train_data, train_transform, args, clean_x, clean_y, True)

    print("Clean Samples: {}".format(clean_x.shape[0]))
    print("Sample Per Class: {}".format(get_noisy_imbalanced_per_cls_list(clean_y, 10)))

    kk_array = torch.zeros(10)
    for cx in clean_x:
        if noisy_labels[cx] == _clean_labels[cx]:
            kk_array[noisy_labels[cx]] += 1
    
    # print(kk_array)
    print(', '.join('{:.4f}'.format(aaa/bbb) for aaa, bbb in zip(kk_array, get_noisy_imbalanced_per_cls_list(clean_y, 10))))

    #train final FC
    
    criterion_fc = nn.CrossEntropyLoss(reduction='none')
    
    train_fc(model, eval_loader, optimizer, criterion_fc, epoch, args)

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

print('Best Overall Acc: {}'.format(best_acc))
print('Best Per-Cls Acc: {}'.format(best_per_cls))
print('Imbalanced Factor {}'.format(imb_factor))
print('Noise Ratio: {}'.format(noise_ratio))