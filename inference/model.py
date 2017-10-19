import sys, os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms

from PIL import Image

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

model_weights = 'models/rt_resnet18ly2_va_0.8700564971751412_model_wts.pkl'
model = RtResnet18ly2()
model.load_state_dict(torch.load(model_weights, map_location=lambda storage, loc: storage))
model.train(False)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Scale(224),
                                 transforms.ToTensor(),
                                 normalize])

def predict_class(image_file):
    img_pil = Image.open(image_file)    
    img_tensor = preprocess(img_pil)
    img_tensor.unsqueeze_(0)

    img_variable = Variable(img_tensor)
    fc_out = model(img_variable)
    fc_out = F.softmax(fc_out)

    labels = {0: 'other', 1: 'retail'}

    print('debug***',fc_out.data.numpy())

    preds = fc_out.data.numpy()
    label = int(preds.argmax())
    class_prediction = labels[label]
    probability = round(100*preds[0][label],2)
    print('preds',preds)
    print('label',label)
    print('prob',probability)
    #return class_prediction
    #predictions = [{'label': label, 'description': class_prediction, 'probability': probability * 100.0}]
    predictions = [{'label': label, 'description': class_prediction, 'probability': probability }]
    return predictions


