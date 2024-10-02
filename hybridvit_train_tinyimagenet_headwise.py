from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
from PIL import ImageDraw, ImageFont, Image
import requests
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import torch
import torch.nn as nn
import numpy as np
from datasets import load_metric
import pandas as pd
#from vit_model import *
import torchvision
from torchvision.transforms import ToTensor, transforms
import torch.utils.data as data
from torch.autograd import Variable
from sklearn import metrics
# from hparams import *
import argparse
import collections
from sklearn.metrics import roc_auc_score
import random
import logging
import sys
from timm.models import create_model, safe_model_name, resume_checkpoint, \
    convert_splitbn_model, model_parameters

import cct
from timm.loss import LabelSmoothingCrossEntropy
import os
from time import time
import argparse
import math
from utils.autoaug import CIFAR10Policy
from utils.helpers import load_checkpoint
from utils import TinyImageNet

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616],
        'teacher_model_path': './best_model/best_student_approx_vit_KD_False_pretrain_True_approx_original_dataset_cifar10_imgsize_32_acc94.21.pth',
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
        'teacher_model_path': './best_model/best_student_approx_vit_KD_False_pretrain_True_approx_original_dataset_cifar100_imgsize_32_bs_256_attention_original_arch_3x1_7_acc78.pth',
    },
        'tinyimagenet': {
        'num_classes': 200,
        'img_size': 32,
        'mean': [0.4802, 0.4481, 0.3975],
        'std': [0.2719, 0.2654, 0.2743]
    }
}
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_parser(parser):

    parser.add_argument('--device', type=str, default='gpu')

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['tinyimagenet'],
                        default='tinyimagenet')

    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--kernel_size', default=7, type=int)
    parser.add_argument('--n_conv_layers', default=2, type=int)
    parser.add_argument('--n_attn_layers', default=9, type=int)
    parser.add_argument('--positional_embedding',type=str, default='sine')

    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--input_N_dim', default=196, type=int)
    parser.add_argument('--initcd', type=str2bool, default=False)

    parser.add_argument('--train_batch_used', default=100000, type=int)
    parser.add_argument('--val_batch_used', default=100000, type=int)
    parser.add_argument('--load_pretrain_model', type=str2bool, default=False) # True
    parser.add_argument('--load_progress_bar', type=str2bool, default=True)

    parser.add_argument('--lr', default=6e-4, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--regularization', default=6e-2, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--batch_size', default=256, type=int) # 512
    parser.add_argument('--print_freq', default=100, type=int)


### parameters for NAS (External Attn + Self Attn)
    parser.add_argument('--NAS_epoch', default=300, type=int)
    parser.add_argument('--random_sa_selection', default = False, type = str2bool)
    parser.add_argument('--fix_sa_selection', default = None, nargs='+', type=int)
    parser.add_argument('--alpha_threshold', default=0.5, type=float) # threshold for hybrid attention mask (alpha)
    parser.add_argument('--lam', default=0.1, type=float) # coefficient for hybrid loss
    parser.add_argument('--kappa', default=1.1, type=float) # coefficient for increasing lam
    parser.add_argument('--self_attn_limit', default=3, type=int) # coefficient for increasing lam


### parameters for finetune
    parser.add_argument('--finetune_epoch', default=600, type =int) # finetune epoch

    parser.add_argument('--softmax_approx', type=str, default='original')
    parser.add_argument('--attention_mechanism', type=str, choices=['original', 'externalattention', 'hybrid', 'hybridHeadWise'], default='original') # original
    parser.add_argument('--externalattention_dim', type=int, default=64)
    parser.add_argument('--externalattention_divhead', type=str2bool, default=True)
    parser.add_argument('--apply_knowledge_distillation', type=str2bool, default=False)
    parser.add_argument('--load_teacher_model', type=str2bool, default=False)

    parser.add_argument('--teacher_model_path',
                        type=str,
                        default='./best_model/imdb_best_finetune_bert_512_bs_32_acc94.25_100_epoch.pth')


    return parser


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res

#def load_pretrain(model, path):


#def train(device, trainloader, model, criterion, optimizer, scheduler, epoch, args):
def train_with_KD(device, trainloader, model, criterion, optimizer, epoch, args, teacher_model=None):
    model.train()
    loss_sum, acc1_num_sum = 0, 0
    num_input_sum = 0
    loss_feature = nn.MSELoss()
    #kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss = nn.CrossEntropyLoss()
    for batch_idx, (images, target) in enumerate(trainloader):
        if batch_idx>args.train_batch_used:
            break

        images, target = images.to(device), target.to(device)

        output, last_vit_feature = model(images)
        #print (f"output: {output.shape}")

        if args.apply_knowledge_distillation:
            teacher_output, teacher_last_vit_feature=teacher_model(images)
            #print (f"teacher_output: {teacher_output.shape}")


        loss = criterion(output, target)

        if args.apply_knowledge_distillation:
            #print (f"original loss:{loss}")
            #print (f"kl_loss: {kl_loss(output, teacher_output)}")
            #print (f"loss_feature: {loss_feature(last_vit_feature, teacher_last_vit_feature)}")
            #loss=loss+kl_loss(output, teacher_output)+loss_feature(last_vit_feature, teacher_last_vit_feature)
            #loss = loss + kl_loss(output, teacher_output) + loss_feature(last_vit_feature, teacher_last_vit_feature)
            loss = loss  + loss_feature(last_vit_feature, teacher_last_vit_feature)
            #print (loss)

        #print (f"output: {output.shape}")

        acc1 = accuracy(output, target)
        num_input_sum += images.shape[0]
        loss_sum += float(loss.item() * images.shape[0])
        acc1_num_sum += float(acc1[0] * images.shape[0])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #scheduler.step()


        if batch_idx % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc1_num_sum / num_input_sum)
            print(f'[Epoch {epoch + 1}][Train][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


def train_hybrid_attention_with_KD(device, trainloader, model, criterion, optimizer, epoch, args, teacher_model=None, lam=0.00001):
    model.train()
    loss_sum, acc1_num_sum = 0, 0
    num_input_sum = 0
    loss_feature = nn.MSELoss()
    #kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss = nn.CrossEntropyLoss()
    for batch_idx, (images, target) in enumerate(trainloader):
        if batch_idx>args.train_batch_used:
            break

        images, target = images.to(device), target.to(device)

        output, last_vit_feature = model(images)
        #print (f"output: {output.shape}")

        if args.apply_knowledge_distillation:
            teacher_output, teacher_last_vit_feature=teacher_model(images)
            #print (f"teacher_output: {teacher_output.shape}")


        loss = criterion(output, target)

        if args.apply_knowledge_distillation:
            #print (f"original loss:{loss}")
            #print (f"kl_loss: {kl_loss(output, teacher_output)}")
            #print (f"loss_feature: {loss_feature(last_vit_feature, teacher_last_vit_feature)}")
            #loss=loss+kl_loss(output, teacher_output)+loss_feature(last_vit_feature, teacher_last_vit_feature)
            #loss = loss + kl_loss(output, teacher_output) + loss_feature(last_vit_feature, teacher_last_vit_feature)
            loss = loss  + loss_feature(last_vit_feature, teacher_last_vit_feature)
            #print (loss)

        #print (f"output: {output.shape}")


        sum_alpha = 0
        for name, param in model.named_parameters():
            if 'alphas' in name:
                sum_alpha += torch.sum(param.data)

        loss = loss + lam * sum_alpha


        acc1 = accuracy(output, target)
        num_input_sum += images.shape[0]
        loss_sum += float(loss.item() * images.shape[0])
        acc1_num_sum += float(acc1[0] * images.shape[0])

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        #scheduler.step()


        if batch_idx % args.print_freq == 0:

            for name, param in model.named_parameters():
                if 'alphas' in name:
                    print(name, param.data, param.requires_grad)
            print('total loss:', loss.data, 'loss_alpha: ', sum_alpha.data, 'lambda: ', lam)

            avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc1_num_sum / num_input_sum)
            print(f'[Epoch {epoch + 1}][Train][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')



def validate(device, testloader, model, criterion, epoch, args, time_begin):
    model.eval()
    loss_sum, acc_num_sum = 0, 0
    num_input_sum = 0
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(testloader):
            if batch_idx > args.val_batch_used:
                break

            images, target = images.to(device), target.to(device)

            output, _ = model(images)
            loss = criterion(output, target)

            acc1 = accuracy(output, target)
            num_input_sum += images.shape[0]
            loss_sum += float(loss.item() * images.shape[0])
            acc_num_sum += float(acc1[0] * images.shape[0])

            if batch_idx % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
                print(f'[Epoch {epoch + 1}][Eval][{batch_idx}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


    avg_loss, avg_acc = (loss_sum / num_input_sum), (acc_num_sum / num_input_sum)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc:6.2f} \t \t Time: {total_mins:.2f}')

    if epoch +1 ==600:
        print('-----alpha check------')
        for name, param in model.named_parameters():
            if 'alphas' in name:
                print(name, param, param.requires_grad)

    return avg_acc

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    #elif not args.disable_cos:
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_self_attn(model, args):
    self_attn_count = 0
    for name, param in model.named_parameters():
        if 'alphas' in name:
            cur_alpha = param.data
            cur_SA_count = torch.numel(cur_alpha[cur_alpha>args.alpha_threshold])
            self_attn_count += cur_SA_count
    return self_attn_count


def loadDataset(args):
    if args.dataset in DATASETS:
    #    img_size=DATASETS[args.dataset]['img_size']
        num_classes = DATASETS[args.dataset]['num_classes']
        img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']

    else:

        num_classes = DATASETS[args.dataset]['num_classes']


    img_size = args.img_size

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    augmentations = []
    augmentations += [
        CIFAR10Policy()
    ]
    augmentations += [
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=(img_size // 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ]

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        *normalize,
    ])
    augmentations = transforms.Compose(augmentations)
    """
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=(img_size // 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(tuple(img_mean), tuple(img_std)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(tuple(img_mean), tuple(img_std)),
    ])
    """

    if args.dataset == 'cifar10':
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=augmentations)
        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)

    elif args.dataset == 'cifar100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(
            root='../data', train=True, download=True, transform=augmentations)
        testset = torchvision.datasets.CIFAR100(
            root='../data', train=False, download=True, transform=transform_test)

    elif args.dataset == 'tinyimagenet':
        num_classes = 200
        data_dir = '../data/tiny-imagenet-200/'
        trainset = TinyImageNet.TinyImageNet(data_dir, train=True, transform=augmentations)
        testset = TinyImageNet.TinyImageNet(data_dir, train=False, transform=transform_test)

    else:
        print('Please use cifar10 or cifar100 dataset.')

    #print (f"Size of trainset: {len(trainset)}")
    #print (f"Size of testset: {len(testset)}")


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    print (f"Size of trainloader: {len(trainloader)}")
    print (f"Size of testloader: {len(testloader)}")
    return trainloader, testloader

def train_top_module(args):
    set_seed(42)

    global a_logger

    a_logger = logging.getLogger()
    a_logger.setLevel(logging.DEBUG)

    output_file_handler = logging.FileHandler(
        f"output_{args.softmax_approx}__KD{args.apply_knowledge_distillation}_{args.dataset}.log")

    stdout_handler = logging.StreamHandler(sys.stdout)
    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)

    if args.device=='gpu' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    print (f'config: {args}')


    trainloader, testloader = loadDataset(args)

    img_size = args.img_size
    if args.dataset in DATASETS:
    #    img_size=DATASETS[args.dataset]['img_size']
        num_classes = DATASETS[args.dataset]['num_classes']
    else:

        num_classes = DATASETS[args.dataset]['num_classes']

    if args.n_attn_layers==9:
        model_name = f'cct_{args.n_attn_layers}_{args.kernel_size}x{args.n_conv_layers}_{img_size}_{args.positional_embedding}'
        print (f"loading model: {model_name}")

        #model=cct_7_7x2_224_sine(pretrained=args.load_pretrain_model,
        model=cct.cct_9_7x2_224_sine(
                            pretrained=args.load_pretrain_model,
                            progress=args.load_progress_bar,
                            #num_layers=args.n_attn_layers,
                            num_heads=args.num_heads,
                            #mlp_ratio=2,
                            #embedding_dim=192,
                            img_size=img_size,
                            positional_embedding=args.positional_embedding,
                            num_classes=num_classes,
                            softmax_approx=args.softmax_approx,
                            attention_mechanism=args.attention_mechanism,
                            externalattention_dim=args.externalattention_dim,
                            input_N_dim=args.input_N_dim,
                            externalattention_divhead=args.externalattention_divhead,
                            # alpha_threshold=args.alpha_threshold,
                            allargs=args
                            )
    elif args.n_attn_layers==7 and img_size==224:
        #model_name = f'cct_14_{args.kernel_size}x{args.n_conv_layers}_{img_size}_{args.positional_embedding}'
        model_name = f'cct_7_{args.kernel_size}x{args.n_conv_layers}_{img_size}_{args.positional_embedding}'
        print (f"loading model: {model_name}")

        model=cct.cct_7_7x2_224_sine(pretrained=args.load_pretrain_model,
        #model=cct.cct_14(model_name,
        #model = cct.cct_7(model_name,
                            #pretrained=args.load_pretrain_model,
                            progress=args.load_progress_bar,
                            img_size=img_size,
                            positional_embedding=args.positional_embedding,
                            num_classes=num_classes,
                            #kernel_size=args.kernel_size,
                            softmax_approx=args.softmax_approx,
                            attention_mechanism=args.attention_mechanism,
                            externalattention_dim=args.externalattention_dim,
                        #  input_N_dim=args.input_N_dim,
                            externalattention_divhead=args.externalattention_divhead,
                            num_heads=args.num_heads,
                            # alpha_threshold=args.alpha_threshold,
                            allargs=args
                            )

    elif args.n_attn_layers==14 and img_size==224:
        #model_name = f'cct_14_{args.kernel_size}x{args.n_conv_layers}_{img_size}_{args.positional_embedding}'
        model_name = f'cct_{args.n_attn_layers}_{args.kernel_size}x{args.n_conv_layers}_{img_size}'
        print (f"loading model: {model_name}")

        model=cct.cct_14_7x2_224(pretrained=args.load_pretrain_model,
        #model=cct.cct_14(model_name,
        #model = cct.cct_7(model_name,
                            #pretrained=args.load_pretrain_model,
                            progress=args.load_progress_bar,
                            img_size=img_size,
                            positional_embedding=args.positional_embedding,
                            num_classes=num_classes,
                         #kernel_size=args.kernel_size,
                            softmax_approx=args.softmax_approx,
                            attention_mechanism=args.attention_mechanism,
                            externalattention_dim=args.externalattention_dim,
                            externalattention_divhead=args.externalattention_divhead,
                            # alpha_threshold=args.alpha_threshold,
                            num_heads=args.num_heads,
                            allargs=args
                            )

    model.to(device)

    teacher_model=None
    if args.apply_knowledge_distillation:
        if args.n_attn_layers==7:
            teacher_model_name = f'cct_7_{img_size}_{args.positional_embedding}'

            teacher_model=cct.cct_7(teacher_model_name,
                            pretrained=False,
                            progress=args.load_progress_bar,
                            img_size=img_size,
                            positional_embedding=args.positional_embedding,
                            num_classes=num_classes,
                            softmax_approx='original',
                            attention_mechanism='original',
                        externalattention_dim=args.externalattention_dim,
                        externalattention_divhead=args.externalattention_divhead,
                            )
        elif args.n_attn_layers==14:
            teacher_model_name = f'cct_14_{img_size}_{args.positional_embedding}'
            #print (f"loading model: {teacher_model_name}")

            # model=cct_7_7x2_224_sine(pretrained=args.load_pretrain_model,
            teacher_model = cct.cct_14(teacher_model_name,
                               pretrained=False,
                               progress=args.load_progress_bar,
                               img_size=img_size,
                               positional_embedding=args.positional_embedding,
                               num_classes=num_classes,
                               softmax_approx='original',
                               attention_mechanism='original',
                               externalattention_dim=args.externalattention_dim,
                               externalattention_divhead=args.externalattention_divhead,
                               )


        teacher_model.to(device)
        teacher_model_path = DATASETS[args.dataset]['teacher_model_path']
        teach_model_state_dict, teach_optimizer_state_dict = load_checkpoint(teacher_model_path)
        teacher_model.load_state_dict(teach_model_state_dict)



    if not os.path.exists('./best_model'):
        os.makedirs('./best_model')



    time_begin = time()
    best_val_acc = 0

    random_sa_selection = args.random_sa_selection
    fix_sa_selection = args.fix_sa_selection

    if random_sa_selection == False and fix_sa_selection == None:
            # NAS is enabled

        ######################## train ViT and attn mask simultaneously (NAS) ####################
        NASepochs = args.NAS_epoch
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.regularization)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.warmup_epoch)
        criterion = LabelSmoothingCrossEntropy()
        criterion.to(device)

        lam = args.lam
        for epoch in range(NASepochs):
            adjust_learning_rate(optimizer, epoch, args)
            #train(device, trainloader, model, criterion, optimizer, scheduler, epoch, args)

            print('----- NAS epoch -------')
            train_hybrid_attention_with_KD(device, trainloader, model, criterion, optimizer, epoch, args, teacher_model=teacher_model, lam=lam)
            
            val_acc=validate(device, testloader, model, criterion, epoch, args, time_begin)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # torch.save(model, f"./best_model/best_student_approx_vit_KD_{args.apply_knowledge_distillation}_pretrain_{args.load_pretrain_model}_approx_{args.softmax_approx}_dataset_{args.dataset}_imgsize_{img_size}_bs_{args.batch_size}_attention_{args.attention_mechanism}_arch_{args.kernel_size}x{args.n_conv_layers}_{args.n_attn_layers}.pth")

            # cur_self_attn_count = count_self_attn(model, args)

            # print('Self Attn count: ', cur_self_attn_count, 'self attn limit: ', args.self_attn_limit, 'lambda: ', lam, 'lr: ', args.lr)

            # if cur_self_attn_count >= args.self_attn_limit:
            #     print('Finish NAS at epoch:', epoch)
            #     break

        print (f'config: {args}')
        total_mins = (time() - time_begin) / 60
        print(f'Script finished in {total_mins:.2f} minutes, '
            f'best top-1: {best_val_acc:.2f}, '
            f'final top-1: {val_acc:.2f}, '
            f'best epoch: {best_epoch}')

        ######################## Freeze alpha and Finetune ####################

        print('------------- Freeze alpha -------------')

        # if there are more self-attn than self_attn_limit, choose the ones with top alphas.
        i = 0
        big_alpha = torch.zeros(args.n_attn_layers, 1, 1, args.num_heads, 1)
        for name, param in model.named_parameters():
            if 'alphas' in name:
                cur_data = param.data
                big_alpha[i] = cur_data
                i += 1


        big_alpha = big_alpha.reshape(1, args.n_attn_layers*args.num_heads)
        _, indices = torch.topk(big_alpha, args.self_attn_limit)
        for indx in indices:
            big_alpha[0,indx] = 1

        big_alpha=torch.where(big_alpha==1, 1.0, 0.0)

        big_alpha = big_alpha.reshape(args.n_attn_layers, 1, 1, args.num_heads, 1)

        i=0
        for name, param in model.named_parameters():
            if 'alphas' in name:
                param.data = big_alpha[i]
                param.requires_grad = False
                param.grad = None
                i += 1

                print(name, param)
    

    elif random_sa_selection == True and fix_sa_selection == None: 
         # random insert self-attention
        sa_location = np.random.randint(args.n_attn_layers, size=args.self_attn_limit)
        print(sa_location)
        sa_name = ['classifier.blocks.{}.self_attn.alphas'.format(i) for i in sa_location]
        print(sa_name)
        for name, param in model.named_parameters():
            if 'alphas' in name:
                if name in sa_name:
                    param.data = torch.ones(1)
                    param.requires_grad = False
                    param.grad = None
                else:
                    param.data = torch.zeros(1)
                    param.requires_grad = False
                    param.grad = None
                print(name, param)       
    
    elif fix_sa_selection != None:
        sa_location = fix_sa_selection
        print(sa_location)
        sa_name = ['classifier.blocks.{}.self_attn.alphas'.format(i) for i in sa_location]
        print(sa_name)
        for name, param in model.named_parameters():
            if 'alphas' in name:
                if name in sa_name:
                    param.data = torch.ones(1)
                    param.requires_grad = False
                    param.grad = None
                else:
                    param.data = torch.zeros(1)
                    param.requires_grad = False
                    param.grad = None
                print(name, param)   

    model.to(device)

    print('------------- Begin Finetune -------------')
    best_val_acc = 0
    best_epoch = 0
    finetune_epoch = args.finetune_epoch
    # reset optimizer for freezed alphas
    optimizer_finetune = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.regularization)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=args.warmup_epoch)

    criterion_finetune = LabelSmoothingCrossEntropy()
    criterion_finetune.to(device)
    best_epoch = 0
    for epoch in range(finetune_epoch):
        adjust_learning_rate(optimizer_finetune, epoch, args)
        train_with_KD(device, trainloader, model, criterion_finetune, optimizer_finetune, epoch, args, teacher_model=teacher_model)
        
        val_acc=validate(device, testloader, model, criterion_finetune, epoch, args, time_begin)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # torch.save(model, f"./best_model/best_student_approx_vit_KD_{args.apply_knowledge_distillation}_pretrain_{args.load_pretrain_model}_approx_{args.softmax_approx}_dataset_{args.dataset}_imgsize_{img_size}_bs_{args.batch_size}_attention_{args.attention_mechanism}_arch_{args.kernel_size}x{args.n_conv_layers}_{args.n_attn_layers}_self_attn_limit_{args.self_attn_limit}.pth")
            torch.save(model, f"./best_model/vit_approx_{args.softmax_approx}_dataset_{args.dataset}_attention_{args.attention_mechanism}_self_attn_limit_{args.self_attn_limit}_total_attn_{args.n_attn_layers}_randomSAselect_{random_sa_selection}_fixSALocation_{fix_sa_selection}_NASepoch_{args.NAS_epoch}_numHead_{args.num_heads}.pth")

        print('current best val: ', best_val_acc, 'best epoch: ', best_epoch)

    print (f'config: {args}')
    total_mins = (time() - time_begin) / 60
    print(f'Script finished in {total_mins:.2f} minutes, '
          f'best top-1: {best_val_acc:.2f}, '
          f'final top-1: {val_acc:.2f}, '
        f'final best epoch: {best_epoch}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CIFAR quick training script')

    parser=init_parser(parser)

    args = parser.parse_args()

    #hparams = PARAMS
    #hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)


    train_top_module(args)


