#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models import efficientnet_b0
from torchvision.models import efficientnet
# from torchsummary import summary
import efficientnet_pytorch

def EfficientNet(args):
    
    model = efficientnet_b0(pretrained=True)
    model.classifier[1].out_features = args.num_classes
    
    return model

# def InceptionNetV3(args):
    
#     model = torchvision.models.inception_v3(pretrained=True)
#     model.classifier[1].out_features = args.num_classes
#     model.aux_logit=False

#     return model

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


# other paper's modified EfficientNet b0 model

class Mod_EfficientNet_b0(nn.Module):

    def __init__(self):
        super(Mod_EfficientNet_b0, self).__init__()
        
        #  super(EfficientNet_b0, self).__init__() is used to inherit nn.Module used above.
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')

        # modified layer adapted from paper
        # self.classifier_layer = nn.Sequential(
            # nn.AvgPool2d(1), # giving issues, removing for now, may not be needed for model
            # nn.Linear(1280 , 512),
            # nn.SiLU(),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.5),
            # nn.Linear(512 , 256),
            # nn.SiLU(),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.5),
            # nn.Linear(256 , 9)
            # )
        self.linear_1 = nn.Linear(1280 , 512)
        self.bn_dro_1 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            )
        self.linear_2 = nn.Linear(512 , 256)
        self.SiLU = nn.SiLU()
        self.bn_dro_2 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            )
        self.classifier_layer = nn.Linear(256 , 9)

    # forward function of Efficient-Net model 
    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        # x = self.classifier_layer(x)
        x = self.linear_1(x)
        x = self.SiLU(x)
        x = self.bn_dro_1(x)
        x = self.linear_2(x)
        x = self.SiLU(x)
        x = self.bn_dro_2(x)
        x = self.classifier_layer(x)
        
        return x


def custom_EN_b0():
    
    model = Mod_EfficientNet_b0() #pretrained = True)
    # model.classifier[1].out_features = args.num_classes
    # summary(ENb0, (3,224,224))
    return model




#################################################################
# our version which outperforms above version from other paper
#################################################################

# this code block is super messy with commented out code but will be cleaned up as soon as some testing is done to figure out what is not required

class Mod_EfficientNet_b0_v2(nn.Module):

    def __init__(self):
        super(Mod_EfficientNet_b0_v2, self).__init__()
        
        #  super(EfficientNet_b0, self).__init__() is used to inherit nn.Module used above.
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        # self.swish_beta = 
        self.conv1x1 = nn.Conv2d(1280, 1024, 1)
        # self.linear_1 = nn.Linear(1024 , 640)
        # self.dro_1 = nn.Sequential(
        #     # nn.BatchNorm1d(640),
        #     nn.Dropout(0.8), # alt 7
        #     )
        # self.dro_1 = nn.Dropout(0.8) # alt 7

        # self.linear_2 = nn.Linear(640 , 400)
        # self.swish = nn.SiLU()
        # self.dro_2 = nn.Dropout(0.5) # alt 4.375

        # self.linear_3 = nn.Linear(400 , 250)
        # self.dro_3 = nn.Dropout(0.3), # alt 2.73

        # self.classifier_layer = nn.Linear(250 , 9)

        self.classifier_layer = nn.Sequential(
            # nn.Conv2d(16, 1024, 1),
            # nn.AvgPool2d(1), # giving issues, removing for now, may not be needed for model
            nn.Linear(1024 , 640),
            nn.SiLU(),
            # nn.BatchNorm1d(512),
            nn.Dropout(0.8),
            nn.Linear(640 , 400),
            nn.SiLU(),
            # nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(400, 250),
            nn.SiLU(),
            # nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(250 , 9)
            )


    # forward function of Efficient-Net model 
    def forward(self, inputs):
        x = self.model.extract_features(inputs)
        x = self.model._avg_pooling(x)
        x = self.conv1x1(x)
        x = x.flatten(start_dim = 1)
        # x = self.model._dropout(x)
        # x = self.linear_1(x)
        # x = self.swish(x)
        # x = self.bn_dro_1(x)
        # x = self.linear_2(x)
        # x = self.swish(x)
        # x = self.bn_dro_2(x)
        # x = self.linear_3(x)
        # x = self.swish(x)
        # x = self.bn_dro_3(x)        
        x = self.classifier_layer(x)
        
        return x

def custom_EN_b0_v2(args):
    
    model = Mod_EfficientNet_b0_v2() #pretrained = True)
    # model.classifier[1].out_features = args.num_classes
    # summary(ENb0, (3,224,224))
    return model
