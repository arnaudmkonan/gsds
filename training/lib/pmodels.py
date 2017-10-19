import sys, os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision



class RtResnet18ly2(nn.Module):
    ''' re-tune Resnet 18 starting from imagenet weights'''
    def __init__(self, num_classes=2):
        super(RtResnet18ly2, self).__init__()
        original_model = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        num_ftrs = original_model.fc.in_features
        self.preclassifier = nn.Sequential(nn.Linear(num_ftrs, 256))
        self.drop = nn.Dropout(0.5)
        self.classifier = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.drop(F.relu(self.preclassifier(f)))
        y = self.classifier(y)
        return y


class FtResnet18(nn.Module):
    ''' fine tune Resnet 18 '''
    def __init__(self, num_classes=2):
        super(FtResnet18, self).__init__()
        original_model = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        # freeze all bottom layers
        for param in self.features.parameters():
            param.requires_grad = False
        num_ftrs = original_model.fc.in_features
        # tune two layers
        self.preclassifier = nn.Sequential(nn.Linear(num_ftrs, 256))
        self.classifier = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = F.relu(self.preclassifier(f))
        y = self.classifier(y)
        return y
