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

class ConvGNReLU(nn.Module):
    def __init__(self, in_size, out_size, group, kernel_size=3, stride=1, padding=1):
        super(ConvGNReLU, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                   nn.GroupNorm(group, out_size),
                                   nn.ReLU(inplace=True),)
        # initialise the blocks
        for m in self.children():
            m.apply(weights_init_kaiming)
        
    def forward(self, inputs):
        return self.conv1(inputs)


class UnetGNConv2D(nn.Module):
    def __init__(self, in_size, out_size, group, kernel_size=3, stride=1, padding=1):
        super(UnetGNConv2D, self).__init__()
        self.conv1 = ConvGNReLU(in_size, out_size, group, kernel_size, stride, padding)
        self.conv2 = ConvGNReLU(out_size, out_size, group, kernel_size, 1, padding)

    def forward(self, inputs):
        x = self.conv1(inputs)
        return self.conv2(x)


class UnetUpGNConv2D(nn.Module):
    def __init__(self, in_size, out_size, group, is_deconv=True):
        super(UnetUpGNConv2D, self).__init__()

        self.conv = UnetGNConv2D(in_size, out_size, group)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetGNConv2D') != -1: 
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset  = output2.size()[2] - input1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(input1, padding)
        output  = torch.cat([output1, output2], 1)
        return self.conv(output)
