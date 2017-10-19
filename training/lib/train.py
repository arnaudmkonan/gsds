import sys, os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

from pmodels import RtResnet18ly2, FtResnet18
from trainer import train_model
from tester import test_model

import argparse
import random


def main():
    '''
    Run training and model saving..see args for options
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsize', help='mini batch size, lower if have memory issues', type=int, default=32)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('--lrs', help='learning rate step decay, ie how many epochs to weight before decaying rate', type=int, default=4)
    parser.add_argument('--lrsg', help='learning rate step decay factor,gamma decay rate', type=float, default=0.1)
    parser.add_argument('--L2', help='L2 weight decay', type=float, default=0.01)
    parser.add_argument('--num_epochs', help='number of epochs', type=int, default=12)
    parser.add_argument('--random_seed', help='use random seed, use 0 for false, 1 for generate, and more than 2 to seed', type=int, default=1)
    parser.add_argument('--model_type', help='retrain or finetune', type=str, default='retrain')
    parser.add_argument('--train_dir', help='train directory in data root', type=str, default='train5')
    parser.add_argument('--model_dir', help='model directory', type=str, default='../data/models/')
    parser.add_argument('--val_dir', help='validation directory in data root', type=str, default='val5')
    parser.add_argument('--data_dir', help='data directory', type=str, default='../data')
    parser.add_argument('--print_class_results', dest='print_class_results', action='store_true')
    parser.add_argument('--no_print_class_results', dest='print_class_results', action='store_false')
    parser.add_argument('--print_batches', dest='print_batches', action='store_true')
    parser.add_argument('--no_print_batches', dest='print_batches', action='store_false')
    parser.set_defaults(print_class_results=True)
    parser.set_defaults(print_batches=True)
    # parse the args 
    args = parser.parse_args()

    print('Settings for training:', 'batch size:', args.bsize, 'epochs:', args.num_epochs, 'learning rate:', args.learning_rate, 'lr decay', args.lrs, 'gamma', args.lrsg)

    if args.random_seed == 1:
        random_seed = random.randint(1,1000)
        print('Random seed:',random_seed)
        # CPU seed
        torch.manual_seed(random_seed)
        # GPU seed
        torch.cuda.manual_seed_all(random_seed)
    else:
        random_seed = args.random_seed

    use_gpu = torch.cuda.is_available()

    data_transforms = { 'train': transforms.Compose([
                           transforms.Scale(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                        ]),
                        'val': transforms.Compose([
                           transforms.Scale(224),
                           transforms.ToTensor(),
                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                        ]),
                      }

    image_datasets = {'train': 
                        datasets.ImageFolder(os.path.join(args.data_dir,args.train_dir),
                                         data_transforms['train']),
                      'val':
                         datasets.ImageFolder(os.path.join(args.data_dir, args.val_dir),
                                         data_transforms['val']),
                      'test':
                         datasets.ImageFolder(os.path.join(args.data_dir, 'test'),
                                         data_transforms['val']),
                     }


    if use_gpu:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.bsize,
                                             shuffle=True, num_workers=8,
                                             pin_memory=True)
                       for x in ['train', 'val','test']}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.bsize,
                                             shuffle=True, num_workers=8)
                       for x in ['train', 'val', 'test']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    batch_frequency = 100
    # assume batch sizes are the same
    print_sizes = {x: len(image_datasets[x])//(args.bsize*batch_frequency) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    nb_classes = len(class_names)

    print('Data set sizes:', dataset_sizes)
    print('Class names:', class_names)
    print('Total classes:', nb_classes)

    if args.model_type == 'retrain':
        model_conv = RtResnet18ly2(nb_classes)
        model_name = 'rt_resnet18ly2'
        print('Model name:', model_name)
        # optimize all parameters when we retrain
        optimizer_conv = optim.Adam(model_conv.parameters(), lr=args.learning_rate, weight_decay=args.L2)
    elif args.model_type == 'finetune':
        model_conv = FtResnet18(nb_classes)
        model_name = 'ft_resnet18'
        print('Model name:', model_name)
        # optimize only the last layers when we fine tune
        optimizer_conv = optim.Adam(list(model_conv.preclassifier.parameters()) +
                                    list(model_conv.classifier.parameters()), lr=args.learning_rate)
    else:
        sys.exit('Error check model type')

    if use_gpu:
        model_conv = model_conv.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # Decay LR by a factor of lrsg (eg 0.1) every lrs epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=args.lrs, gamma=args.lrsg)

    model_conv, val_acc = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                                  class_names, args.bsize, args.model_dir, model_name, print_sizes,
		                          data_transforms, image_datasets, dataloaders, dataset_sizes,
                                  use_gpu, args.num_epochs, args.print_class_results, args.print_batches)

    # evaluate test set
    test_model(model_conv, criterion, class_names, args.bsize, args.model_dir, model_name, print_sizes,
           dataloaders, dataset_sizes, use_gpu, True)

    # write out best model to disk
    val_acc = round(100*val_acc,1)
    torch.save(model_conv.state_dict(), args.model_dir + model_name +
                                    '_va_' + str(val_acc) +'_model_wts.pth')

    return


if __name__ == '__main__':
    main()
