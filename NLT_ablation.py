from datetime import datetime
from torch.utils.data import DataLoader
from resnet_cifar import *
from tqdm import tqdm
import time
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
import dataloader_cifar as dataloader
from torch.nn.parameter import Parameter
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
import copy
from sampler import *


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--arch', default='ResNet18')
parser.add_argument('--dataset', default='cifar10', help='dataset')
parser.add_argument('--hidden_size', default=512, help='hidden_size')

parser.add_argument('--e_lr', default=0.01, type=float, help='initial learning rate for encoder', dest='e_lr')
parser.add_argument('--fc_lr', default=1, type=float, help='initial learning rate for fc', dest='fc_lr')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--use_mixup', default=False, type=bool, help='use mixup')
parser.add_argument('--use_cutmix', default=False, type=bool, help='use cutmix')
parser.add_argument('--data_split', default='imb', type=str, help='data split type')

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--results-dir', default='', type=str, help='path to cache (default: none)')
parser.add_argument('--beta_flag', default=None, type=str, help='beta')
parser.add_argument('--lam', default=0, type=float, help='ema')


parser.add_argument('--imb_factor', default=0.1, type=float, help='imbalance factor')
parser.add_argument('--noise_ratio', default=0.1, type=float, help='initial learning rate for encoder')



args = parser.parse_args()  # running in ipynb

# set command line arguments here when running in ipynb
args.epochs = 100
args.cos = False
args.schedule = []  # cos in use
args.symmetric = False

ot_criterion = SinkhornDistance_uniform(eps=0.1, max_iter=200, reduction='mean').to('cuda')
# ot_criterion_q = SinkhornDistance(eps=0.1, max_iter=200, reduction='mean').to('cuda')

if args.results_dir == '':
    args.results_dir = './debug/Supervised-Ablation-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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


def train_2(net, data_loader, train_optimizer, criterion, scheduler, epoch, avg_prototypes, args, prior_weights):

    loss, train_bar = 0.0, tqdm(data_loader)
    total = 0.0
    correct = 0.0
    total_2 = 0.0

    total_p_true = 0
    total_p_equals_given = 0

    true_y = []
    true_x = []
    true_y_prob = []

    estimated_all_y = []
    all_x_index = []

    for imgs, labels, index in train_bar:
        train_optimizer.zero_grad()

        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
 
        net.encoder.eval()
        net.classifier.eval()
        feats, _ = net(imgs)

        consistency_loss, ot_pseudo_labels, consistency_index, p_prob = OT_consistency_2(feats, labels, avg_prototypes, prior_weights=prior_weights)

        # consistency_index = torch.ones_like(consistency_index).cuda()
        true_index = torch.nonzero(consistency_index).squeeze()
        true_label = ot_pseudo_labels[true_index]
        all_x_index.append(index)
        true_x_index = index[true_index]
        true_label_prob = p_prob[true_index]


        true_x.append(true_x_index.cpu().detach())
        true_y.append(true_label.cpu().detach())
        true_y_prob.append(true_label_prob.cpu().detach())
        estimated_all_y.append(ot_pseudo_labels.cpu().detach())

        if len(true_index.shape) == 0 or true_index.shape[0] <= 3:
            continue

        net.encoder.train()
        net.classifier.train()
        
        inputs, _labels = pad_to_multiple_of_8(imgs[true_index], true_label)

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
    estimated_all_y = torch.concat(estimated_all_y).cpu().numpy()
    all_x_index = torch.concat(all_x_index).cpu().numpy()

    return loss.item(), [true_x, true_y, true_y_prob], [estimated_all_y, all_x_index]


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

conf_penalty = NegEntropy()

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
    

def test(net, data_loader, criterion, epoch, args, avg_prototypes):
    net.eval()
    correct = 0
    correct_ot = 0

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
   
            correct_ot += ot_pseudo_labels.eq(labels).cpu().sum().item()                 

            _, predicted = torch.max(outputs, 1)            
            correct += predicted.eq(labels).cpu().sum().item()                 

            total_preds.append(predicted)
            total_targets.append(labels)

            test_loss = criterion(outputs, labels)
            total_test_loss.append(test_loss)
            
            total += imgs.size(0)

            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% OT-Acc@1:{:.2f}%'.format(epoch, args.epochs, correct / total * 100, correct_ot / total * 100))

    acc = 100.*correct/total

    acc_ot = 100.*correct_ot/total

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    cls_acc = [ round( 100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) \
                for i in range(args.num_classes)]
    
    return acc, cls_acc, acc_ot


def update_eval_loader(train_data, transform, args, index_x, labels, drop_last=False):

    eval_train_dataset = CustomTensorDataset(train_data[index_x], labels, transform=transform)

    class_idxs = get_idxs_per_cls_list(labels, args.num_classes)

    eval_loader = DataLoader(
            eval_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=12,
        )
    
    return eval_loader


def warmup(epoch, net, optimizer, dataloader, criterion):
    net.train()
    
    loss, train_bar = 0.0, tqdm(dataloader)
    total = 0.0
    correct = 0.0

    for imgs, labels, index in train_bar:
        imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        
        feats, outputs = net(imgs)
           
        loss = criterion(outputs, labels).mean()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        
        loss.backward()  
        optimizer.step() 

        train_bar.set_description('Warming-up Epoch: [{}/{}], Train Acc: {:.6f}, Loss-1: {:.4f}'.format(
                                epoch, 100, 100.*correct/total, loss.item()))

if args.dataset == 'cifar10':
    loader = dataloader.cifar_dataloader('cifar10', imb_type='exp', imb_factor=args.imb_factor, noise_mode=args.data_split, noise_ratio=args.noise_ratio,\
        batch_size=args.batch_size, num_workers=12, root_dir='./data/cifar-10-batches-py', log='log.txt')
else:
    loader = dataloader.cifar_dataloader('cifar100', imb_type='exp', imb_factor=args.imb_factor, noise_mode=args.data_split, noise_ratio=args.noise_ratio,\
        batch_size=args.batch_size, num_workers=12, root_dir='./data/cifar-100-python', log='log.txt')

train_data = torch.Tensor(loader.run('warmup').dataset.train_data).cpu().numpy()
clean_labels = torch.Tensor(loader.run('warmup').dataset.clean_targets).long()
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()


train_dataset = CustomTensorDataset(train_data, noisy_labels, transform=train_transform)

args.num_classes = torch.unique(noisy_labels).size(0)

pcl = get_noisy_imbalanced_per_cls_list(noisy_labels, args.num_classes)
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

args.resume = "cifar{}_IF{}_{}_moco/model_best.pth".format(args.num_classes, int(1/args.imb_factor), args.arch)

saved_ckpt = torch.load(args.resume, 'cuda')
encoder = []
for key in list(saved_ckpt['state_dict'].keys()):
    if key.startswith('encoder_q'):
        _key = key.replace('encoder_q.net.', '')
        encoder.append((_key, saved_ckpt['state_dict'][key]))

if args.arch == 'ResNet32':
    args.hidden_size = 64

model = ModelBase(class_num=args.num_classes, arch=args.arch, resume_ckpt=encoder)
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

# best_warm_acc = 0
# best_warm_model = None

# eval_loader = update_eval_loader(train_data, test_transform, args, index, noisy_labels[index])
# init_avg_prototypes = obtain_noised_features_prototype(model, eval_loader, 0, args)
# avg_prototypes = init_avg_prototypes

# for epoch in range(100):

#     warmup(epoch, model, optimizer, train_loader, criterion)

#     test_acc_1, test_acc_per_cls, test_acc_1_ot = test(model, test_loader, criterion, epoch, args, avg_prototypes)
#     print(test_acc_per_cls)
#     if test_acc_1 > best_warm_acc:
#         best_warm_acc = test_acc_1
#         best_warm_model = copy.deepcopy(model)
    
# print('Best Warm-Acc Acc: {}'.format(best_warm_acc))
# print('Best Warm-OT Acc: {}'.format(test_acc_1_ot))
# print('Best Per-Cls Acc: {}'.format(best_per_cls))
# print('Imbalanced Factor {}'.format(args.imb_factor))
# print('Noise Ratio: {}'.format(args.noise_ratio))
# import pdb;pdb.set_trace()

clean_x = []
clean_y = []
clean_y_prob = []
p_r_a = []
ot_p_r_a = []

sample_per_cls = []
acc_per_cls = []
all_test = []
sample_per_class = []

ot_original_labels = []

eval_loader = update_eval_loader(train_data, test_transform, args, index, noisy_labels[index])
init_avg_prototypes = obtain_noised_features_prototype(model, eval_loader, 0, args)
avg_prototypes = init_avg_prototypes
save_code(__file__, args.results_dir)


if args.beta_flag is not None:
    prior_weights = compute_weights(float(args.beta_flag), get_noisy_imbalanced_per_cls_list(noisy_labels, args.num_classes))
else:
    prior_weights = None

for epoch in range(epoch_start, args.epochs + 1):
    ss = time.time()

    train_loss, real_sample_index, estimated_all = train_2(model, train_loader, optimizer, criterion, scheduler, epoch, avg_prototypes, args, prior_weights)
    results['train_loss'].append(train_loss)

    if args.dataset == 'cifar10':
        clean_x = torch.tensor(real_sample_index[0]).long().numpy()
        clean_y = noisy_labels[clean_x]
        clean_y_prob = real_sample_index[2]
        _cly_p = clean_y_prob

    else:
        clean_x = torch.cat([torch.tensor(clean_x), torch.tensor(real_sample_index[0])], 0)
        clean_y = torch.cat([torch.tensor(clean_y), torch.tensor(real_sample_index[1])], 0)
        clean_y_prob = torch.cat([torch.tensor(clean_y_prob), torch.tensor(real_sample_index[2])], 0)

        _, unique_indices = np.unique(clean_x.numpy(), return_index=True)

        clean_x = clean_x[unique_indices].long().numpy()
        clean_y = clean_y[unique_indices].long()
        clean_y_prob = clean_y_prob[unique_indices]
        _cly_p = clean_y_prob.cpu()

    _cl = _clean_labels[clean_x].cpu()
    _cly = clean_y.cpu()

    try:
        _auc = roc_auc_score(_cl, _cly_p, average='macro', multi_class='ovr')
    except:
        _auc = 0.0

    _precision = precision_score(_cly, _cl, average='macro')
    _recall = recall_score(_cly, _cl, average='macro')
    _acc = accuracy_score(_cly, _cl)

    print("Precision: {:.4f} Recall: {:.4f} Acc: {:.4f} AUC: {:.4f}".format(_precision, _recall, _acc, _auc))
    p_r_a.append([_precision, _recall, _acc, _auc])

    ot_precision = precision_score(estimated_all[0], _clean_labels[estimated_all[1]].cpu(), average='macro')
    ot_recall = recall_score(estimated_all[0], _clean_labels[estimated_all[1]].cpu(), average='macro')
    ot_acc = accuracy_score(estimated_all[0], _clean_labels[estimated_all[1]].cpu())

    print("OT-Precision: {:.4f} OT-Recall: {:.4f} OT-Acc: {:.4f}".format(ot_precision, ot_recall, ot_acc))
    ot_p_r_a.append([ot_precision, ot_recall, ot_acc])

    eval_loader = update_eval_loader(train_data, train_transform, args, clean_x, clean_y, True)

    print("Clean Samples: {}".format(clean_x.shape[0]))

    print("Before Filter Sample Per Class: {}".format(get_noisy_imbalanced_per_cls_list(estimated_all[0], args.num_classes)))
    ot_original_labels.append(get_noisy_imbalanced_per_cls_list(estimated_all[0], args.num_classes))

    print("After Filter Sample Per Class: {}".format(get_noisy_imbalanced_per_cls_list(clean_y, args.num_classes)))

    kk_array = torch.zeros(args.num_classes)
    for cx in clean_x:
        if noisy_labels[cx] == _clean_labels[cx]:
            kk_array[noisy_labels[cx]] += 1
    
    # print(kk_array)
    acc_per_cls_details = ', '.join('{:.4f}'.format(aaa/bbb) for aaa, bbb in zip(kk_array, get_noisy_imbalanced_per_cls_list(clean_y, args.num_classes)))
    print(acc_per_cls_details)
    acc_per_cls.append(acc_per_cls_details)


    criterion_fc = nn.CrossEntropyLoss(reduction='none')    
    train_fc(model, eval_loader, optimizer, criterion_fc, epoch, args)

    tmp_avg_prototypes = obtain_noised_features_prototype(model, eval_loader, 0, args)
    
    if args.dataset == 'cifar10':
        args.lam = 1
    else:
        if args.data_split == 'imb':
            args.lam = 0.9
        else:
            args.lam = 1

    
    avg_prototypes = args.lam * init_avg_prototypes + (1 - args.lam) * tmp_avg_prototypes
    avg_prototypes = tmp_avg_prototypes

    if args.beta_flag is not None:
        prior_weights = compute_weights(float(args.beta_flag), get_noisy_imbalanced_per_cls_list(clean_y, args.num_classes))
    else:
        prior_weights = None


    #evaluate the model    
    test_acc_1, test_acc_per_cls, test_acc_1_ot = test(model, test_loader, criterion, epoch, args, avg_prototypes)
    print(test_acc_per_cls)
    all_test.append(test_acc_per_cls)
    sample_per_class.append(get_noisy_imbalanced_per_cls_list(clean_y, args.num_classes))

    results['test_acc@1'].append(test_acc_1)
    if test_acc_1 > best_acc:
        best_acc = test_acc_1
        best_per_cls = test_acc_per_cls
        best_ot_acc = test_acc_1_ot
        best_ckpt = args.results_dir + '/model_best.pth'
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, best_ckpt)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
    # save model
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')
    ee = time.time()
    print('Time Usage: {}'.format(ee - ss))
    print()


save_meta_train_info(p_r_a, args.results_dir, 'precision_recall_acc.txt')
save_meta_train_info(acc_per_cls, args.results_dir, 'acc_per_cls.txt')
save_meta_train_info(all_test, args.results_dir, 'test_acc_per_cls.txt')
save_meta_train_info(ot_p_r_a, args.results_dir, '{}_{}_{}_OT_precision_recall_acc.txt'.format(args.dataset, int(1/args.imb_factor), args.noise_ratio))
save_meta_train_info(sample_per_class, args.results_dir, '{}_{}_{}_sample_per_class.txt'.format(args.dataset, int(1/args.imb_factor), args.noise_ratio))
save_meta_train_info(ot_original_labels, args.results_dir, '{}_{}_{}_OT_sample_per_class.txt'.format(args.dataset, int(1/args.imb_factor), args.noise_ratio))

print('Best Overall Acc: {}'.format(best_acc))
print('Best OT Acc: {}'.format(best_ot_acc))
print('Best Per-Cls Acc: {}'.format(best_per_cls))
print('Imbalanced Factor {}'.format(args.imb_factor))
print('Noise Ratio: {}'.format(args.noise_ratio))