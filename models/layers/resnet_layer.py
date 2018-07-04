import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.layers.AtrousSpatialPyramidPool import ConvBNRelu

class Bottleneck(nn.Module):
    """Bottleneck Unit"""
    def __init__(self, in_c, mid_c, out_c, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = ConvBNRelu(in_c, mid_c, 1, 
                                 stride=stride, padding=0, dilation=1)
        self.conv3x3 = ConvBNRelu(mid_c, mid_c, 3, 
                                  stride=1, padding=dilation, dilation=dilation)
        self.increase = nn.Sequential(nn.Conv2d(mid_c, out_c, 1, 1, 0, 1),
                                      nn.BatchNorm2d(out_c))

        self.downsample = downsample
        if self.downsample:
            self.proj = nn.Sequential(nn.Conv2d(in_C, out_c, 1, stride=stride, padding=0, dilation=1),
                                      nn.BatchNorm2d(out_c))
    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, n_layers, in_channels, mid_c, out_c, stride, dilation):
        super(_ResBlock, self).__init__()
        blocks = [Bottleneck(in_c, mid_c, out_c, stride, dilation, True)]
        for i in range(2, n_layers + 1):
            blocks.append(Bottleneck(out_c, mid_c, out_c, 1, dilation, False))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResBlockMultiGrid(nn.Module):
    """3x Residual Block with multi-grid"""

    def __init__(self, n_layers, in_channels, mid_c, out_c, stride, dilation, mg=[1, 2, 1]):
        super(ResBlockMultiGrid, self).__init__()
        self.blocks = nn.Sequential(
            Bottleneck(in_channels, mid_channels, out_c, stride, dilation * mg[0], True),
            Bottleneck(out_c, mid_channels, out_c, 1, dilation * mg[1], False),
            Bottleneck(out_c, mid_channels, out_c, 1, dilation * mg[2], False)
        )

    def forward(self, x):
        return self.blocks(x)

