import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Residual(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, group, norm, dilate=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, hidden_c, 1, 1, 0),
                                   norm(in_c),
                                   nn.ReLU(inplace=True),)


        self.conv2 = nn.Sequential(nn.Conv2d(hidden_c, hidden_c, 3, 1, dilate, dilate, group),
                                   norm(in_c),
                                   nn.ReLU(inplace=True),)

        self.conv3 = nn.Sequential(nn.Conv2d(hidden_c, out_c, 1, 1, 0),
                                   norm(out_c),
                                   nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class UnetResConv2D(nn.Module):
    def __init__(self, in_size, out_size, group, norm, kernel_size=3, stride=1, padding=1):
        super(UnetResConv2D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                   norm(out_size),
                                   nn.ReLU(inplace=True),)
        self.res = Residual(out_size, out_size, out_size, group, norm)

    def forward(self, x):
        x = self.conv1(x)
        y = self.res(x)
        return x + y


class UnetResUpConv2D(nn.Module):
    def __init__(self, in_size, out_size, group, norm):
        super(UnetResUpConv2D, self).__init__()

        self.up = nn.ConvTranspose2d(in_size, in_size, 
                                     kernel_size=3, stride=2, padding=1,
                                     output_padding=1)

        self.conv = nn.Sequential(nn.Conv2d(in_size * 2, out_size, 3, 1, 1),
                                  norm(out_size),
                                  nn.ReLU(inplace=True),)
        self.res = Residual(out_size, out_size//2, out_size, group, norm)

    def forward(self, x1, x2):
        output2 = self.up(x2)
        offset  = output2.size()[2] - x1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(x1, padding)
        output  = torch.cat([output1, output2], 1)
        output  = self.conv(output)
        return output + self.res(output)

class SpatialBridge(nn.Module):
    def __init__(self, in_c, dilates, groups, norm):
        super(SpatialBridge, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, in_c * 2, 3, 1, 1),
                                   norm(in_c * 2),
                                   nn.ReLU(inplace=True))

        self.b1 = Residual(in_c * 2, in_c * 2, in_c * 2, groups[0], norm, dilates[0])
               
        self.conv2 = nn.Sequential(nn.Conv2d(in_c * 2, in_c * 4, 3, 1, 1),
                                   norm(in_c * 4),
                                   nn.ReLU(inplace=True))
        self.b2 = Residual(in_c * 4, in_c * 4, in_c * 4, groups[1], norm, dilates[1])

        self.conv3 = nn.Sequential(nn.Conv2d(in_c * 4, in_c * 8, 3, 1, 1),
                                   norm(in_c * 8),
                                   nn.ReLU(inplace=True))
        self.b3 = Residual(in_c * 8, in_c * 8, in_c * 8, groups[2], norm, dilates[2])

        self.cat = nn.Sequential(nn.Conv2d(in_c * 8, in_c, 1, 1, 0),
                                 norm(in_c),
                                 nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv1(x)
        y = self.b1(x)

        x = self.conv2(x + y)
        y = self.b2(x)

        x = self.conv3(x + y)
        y = self.b3(x)
        return self.cat(x)

