import sys, os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision


def test_model(model, criterion, class_names,
               batch_size, model_dir, model_name, print_sizes,
	           dataloaders, dataset_sizes,
	           use_gpu, print_class_results=True):
    ''' test loop '''
    since = time.time()

    nb_classes = len(class_names)
    model.train(False)

    running_loss = 0.0
    running_corrects = 0
    if print_class_results:
        class_correct = list(0. for i in range(nb_classes))
        class_total = list(0. for i in range(nb_classes))

    phase='test'
    # Iterate over test data.
    for data in dataloaders[phase]:
        # get the batch inputs
        inputs_0, labels_0 = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs_0.cuda())
            labels = Variable(labels_0.cuda())
        else:
            inputs, labels = Variable(inputs_0), Variable(labels_0)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # statistics
        batch_loss = loss.data[0]
        batch_correct = torch.sum(preds == labels.data)
        running_loss += batch_loss
        running_corrects += batch_correct

        if print_class_results:
            c = (preds == labels.data).squeeze()
            for i in range(labels_0.size()[0]):
                label = labels_0[i]
                class_correct[label] += c[i]
                class_total[label] += 1

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects / dataset_sizes[phase]

    if print_class_results:
        print()
        for i in range(nb_classes):
            print('%5s accuracy of %7s : %2d %%' % ('Test',
                  class_names[i], 100 * class_correct[i] / class_total[i]))

        print('\n*** Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    print()

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return



