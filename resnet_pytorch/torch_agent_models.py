import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# 残差块
class Residual(nn.Module):  
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        # 当通道数量改变时使用 1 × 1 卷积来匹配通道
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


# 一个残差层,由形状相同的多个残差块构成
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk



class Resnet_18(nn.Module):
    def __init__(self, act_num, softmax=True):
        super().__init__()
        self.softmax = softmax
        self.b1 = nn.Sequential( 
                            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.cov_out = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        )

        self.out_a = nn.Sequential(
        				nn.Linear(512, 256),
        				nn.ReLU(inplace=True),
        				nn.Linear(256, act_num)
                        )
        self.out_c = nn.Sequential(
        				nn.Linear(512, 256),
        				nn.ReLU(inplace=True),
        				nn.Linear(256, 1)
                        )


    def forward(self, x_in):
        x = self.b1(x_in)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.cov_out(x)
        if self.softmax:
            a = F.softmax(self.out_a(x), dim=1)
        else:
            a = self.out_a(x)
        c = self.out_c(x)
        return a, c


class Vgg(nn.Module):
    def __init__(self, act_num, softmax=True):
        super().__init__()
        self.softmax = softmax
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.b5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.cov_out = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        self.out_a = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, act_num)
                        )
        self.out_c = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, 1)
                        )
    def forward(self, x_in):
        x = self.b1(x_in)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.cov_out(x)
        if self.softmax:
            a = F.softmax(self.out_a(x), dim=1)
        else:
            a = self.out_a(x)
        c = self.out_c(x)
        return a, c



if __name__ == '__main__':
    from torchsummary import summary
    model = Resnet_18(act_num=6)
    a = torch.rand([1, 3, 256, 256])
    print(model(a))
    model = model.to("cuda:0")
    summary(model, (3, 256, 256))
    del model
    model = Vgg(act_num=6).to("cuda:0")
    summary(model, (3, 256, 256))