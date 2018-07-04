import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.unet_layer import UnetConv2D, UnetConv3D, weights_init_kaiming

class UnetUpConv2D_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUpConv2D_CT, self).__init__()

        self.conv = UnetConv2D(in_size + out_size, out_size, is_batchnorm, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2D') != -1: 
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset  = output2.size()[2] - input1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output1 = F.pad(input1, padding)
        output  = torch.cat([output1, output2], 1)
        return self.conv(output)


class UnetUpConv3D_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3D(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3D') != -1:
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset  = output2.size()[2] - input1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        output1 = F.pad(input1, padding)
        output  = torch.cat([output1, ouput1], 1)
        return self.conv(ouptut)

class UnetDsv2D(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv2D, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class UnetDsv3D(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3D, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)
