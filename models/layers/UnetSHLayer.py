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

        
class Shortcut(nn.Module):
    def __init__(self, out_size, sh_size, kernel_size=3, stride=1, padding=1):
        super(Shortcut, self).__init__()

        conv = []
        for sh in range(0, sh_size):
            conv.append(nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, stride, padding),
                                      nn.BatchNorm2d(out_size),
                                      nn.ReLU(inplace=True),))
        self.conv = nn.Sequential(*conv)
        self.conv_last = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, stride, padding),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
    
    def forward(self, x):
        res = x + self.conv(x)
        return self.conv_last(x)


class UnetSHConv2D(nn.Module):
    def __init__(self, in_size, out_size, sh_size, kernel_size=3, stride=1, padding=0):
        super(UnetSHConv2D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_size),
                                    nn.ReLU(inplace=True),)
        self.shortcut = Shortcut(out_size, sh_size)

        # initialise the blocks
        for m in self.children():
            m.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shortcut(x)
        return x


class UnetSHUpConv2D(nn.Module):
    def __init__(self, in_size, out_size, sh_size, is_deconv=True):
        super(UnetSHUpConv2D, self).__init__()

        self.conv = UnetSHConv2D(in_size, out_size, sh_size, padding=1)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, 
                                         kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetSHConv2D') != -1: 
                continue
            m.apply(weights_init_kaiming)

    def forward(self, x1, x2):
        output2 = self.up(x2)
        offset  = output2.size()[2] - x1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(x1, padding)
        output  = torch.cat([output1, output2], 1)
        return self.conv(output)
