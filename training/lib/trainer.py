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


def train_model(model, criterion, optimizer, scheduler, class_names,
                batch_size, model_dir, model_name, print_sizes,
		data_transforms, image_datasets, dataloaders, dataset_sizes,
	        use_gpu, num_epochs=25, print_class_results=True, print_batch_results=True):
    ''' training loop
    '''
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = 0
    nb_classes = len(class_names)

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if print_class_results:
                class_correct = list(0. for i in range(nb_classes))
                class_total = list(0. for i in range(nb_classes))

            batch_number = 0
            total_print_batches = 1
            freq = print_sizes[phase]
            bsince = time.time() 
            # Iterate over data.
            for data in dataloaders[phase]:
                if print_batch_results:
                    batch_number += 1
                # get the batch inputs
                inputs_0, labels_0 = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs_0.cuda())
                    labels = Variable(labels_0.cuda())
                else:
                    inputs, labels = Variable(inputs_0), Variable(labels_0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                batch_loss = loss.data[0]
                batch_correct = torch.sum(preds == labels.data)
                running_loss += batch_loss
                running_corrects += batch_correct

                if print_batch_results:
                    if batch_number == freq:
                        batch_losses_avg = running_loss / (batch_size*freq*total_print_batches) 
                        batch_accs = running_corrects / (batch_size*freq*total_print_batches)
                        batch_number = 0
                        btime = time.time() - bsince
                        print('Epoch: {} Frac: {} Phase: {} Batches loss: {:.4f} Batches acc: {:.4f} Time: {:.1f}s'.format(
                              epoch, total_print_batches, phase, batch_losses_avg, batch_accs, btime))
                        total_print_batches +=1
                        bsince = time.time() 

                if print_class_results:
                    c = (preds == labels.data).squeeze()
                    for i in range(labels_0.size()[0]):
                        label = labels_0[i]
                        class_correct[label] += c[i]
                        class_total[label] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                phase_name = 'Train'
            else:
                phase_name = 'Valid'

            if print_class_results:
                print()
                for i in range(nb_classes):
                    print('%5s accuracy of %7s : %2d %%' % (phase_name,
                           class_names[i], 100 * class_correct[i] / class_total[i]))

            print('\n*** {} Epoch Loss: {:.4f} Epoch Acc: {:.4f}'.format(
                  phase_name, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = model.state_dict()
                print('Save model on epoch:', epoch)
                torch.save(best_model_wts, model_dir + model_name + '_model_wts.pkl')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} @epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc



