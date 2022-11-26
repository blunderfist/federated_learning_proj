#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models import efficientnet_b0
from torchvision.models import efficientnet
from torchsummary import summary
# from efficientnet_pytorch import EfficientNet
import efficientnet_pytorch

def EfficientNet(args):
    
    model = efficientnet_b0(pretrained=True)
    model.classifier[1].out_features = args.num_classes
    
    return model

def ResNet(args):
    
    model = torchvision.models.resnet50(pretrained=True)
    model.fc.out_features = args.num_classes
    
    return model

def VGG(args):
    
    model = torchvision.models.vgg19(pretrained=True)
    model.classifier[6].out_features = args.num_classes
    
    return model
    

def EfficientNet_b0(args):
    
    model = efficientnet_b0(pretrained = True)
    model.classifier[1].out_features = args.num_classes
#     # summary(ENb0, (3,224,224))
    return model


# our modified EfficientNet b0 model

class Mod_EfficientNet_b0(nn.Module):

    def __init__(self):
        super(Mod_EfficientNet_b0, self).__init__()
        
        #  super(EfficientNet_b0, self).__init__() is used to inherit nn.Module used above.
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')

        # modified layer adapted from paper
        self.classifier_layer = nn.Sequential(
            # nn.AvgPool2d(1), # giving issues, removing for now, may not be needed for model
            nn.Linear(1280 , 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512 , 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256 , 9)
            )

    # forward function of Efficient-Net model 
    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        # x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x


def custom_EN_b0(args):
    
    model = Mod_EfficientNet_b0() #pretrained = True)
    # model.classifier[1].out_features = args.num_classes
    # summary(ENb0, (3,224,224))
    return model


# custom_EN_b0(9)
# summary(custom_EN_b0(9), (3,224,224))
# print(EN_b0(9))
# print(type(EN_b0(9)) == type(VGG(9)))
# print("custom type: ", type(EN_b0(9)))
# print("vgg19 type: ", type(VGG(9)))