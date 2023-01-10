import torch
import torch.nn as nn
from torch.nn import *
from torchvision import models

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AlexNetFC(nn.Module):
    def __init__(self,n_classes = 8):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer1 = nn.Linear(256,n_classes)

    def forward(self, x):
        # x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer1(pooled_features)
        return output

class VGG11FC(nn.Module):
    def __init__(self,n_classes = 8):
        super().__init__()
        self.pretrained_model = models.vgg11_bn(pretrained=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer1 = nn.Linear(512,n_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer1(pooled_features)
        return output

class SE46_VGG11_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg11_bn = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            SELayer(256),

            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            SELayer(512),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = Dropout(p=0.5, inplace=False)
        self.classifer1 = nn.Linear(512,8)     


    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.vgg11_bn(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer1(pooled_features)
        return output

class SE3456_VGG11_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg11_bn = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            SELayer(256),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            SELayer(256),

            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            SELayer(512),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            SELayer(512),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = Dropout(p=0.5, inplace=False)
        self.classifer1 = nn.Linear(512,8)     


    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.vgg11_bn(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer1(pooled_features)
        return output

class SE234567_VGG11_bn(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg11_bn = nn.Sequential(
            Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            SELayer(128),

            Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            SELayer(256),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            SELayer(256),

            Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            SELayer(512),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            SELayer(512),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            SELayer(512),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = Dropout(p=0.5, inplace=False)
        self.classifer1 = nn.Linear(512,8)     


    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.vgg11_bn(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifer1(pooled_features)
        return output

