import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname == 'Conv2D' != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('GroupNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class ConvNormReLU(nn.Module):
    def __init__(self, in_c, out_c, norm, kernel_size=3, stride=1, padding=1, group=1):
        super(ConvBNReLU, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
                                   norm(out_c),
                                   nn.ReLU(inplace=True),)
        
    def forward(self, inputs):
        return self.conv1(inputs)


class GroupBlock(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, norm, group):
        super(GroupBlock, self).__init__()
        self.conv1 = ConvNormReLU(in_c, hidden_c, norm, 1, 1, 0)
        self.gconv = ConvNormReLU(hidden_c, hidden_c, norm, 3, 1, 0, group)
        self.conv3 = ConvNormReLU(hidden_c, out_c, norm, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gconv(x)
        return self.conv3(x)


class UnetResConv2D(nn.Module):
    def __init__(self, in_c, out_c, norm, group, kernel_size=3, stride=1, padding=1):
        super(UnetResConv2D, self).__init__()
        self.conv1 = ConvNormReLU(in_c, out_c, norm, kernel_size, stride, padding)        
        self.group = GroupBlock(out_c, out_c, out_c, group, norm)

    def forward(self, x):
        x = self.conv1(x)
        y = self.group(x)
        return x + y


class UnetResUpConv2D(nn.Module):
    def __init__(self, in_c, out_c, group, norm):
        super(UnetResUpConv2D, self).__init__()

        self.up = nn.ConvTranspose2d(in_c, in_c, 
                                     kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        self.conv = ConvNormReLU(in_c * 2, out_c, 3, 1, 1)
        self.res = GroupBlock(out_c, out_c//2, out_c, group, norm)

    def forward(self, x1, x2):
        output2 = self.up(x2)
        offset  = output2.size()[2] - x1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(x1, padding)
        output  = torch.cat([output1, output2], 1)
        output  = self.conv(output)
        return output + self.res(output)

