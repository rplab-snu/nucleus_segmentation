import torch
import torch.nn as nn

class ConvBNRelu(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=0, dilation=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, input_):
        return self.conv(input_)


class AtrousSpatialPyramidPool2D(nn.Module):
    def __init__(self, in_c, out_c, dilation_rate=(2, 4, 8)):
        super(AtrousSpatialPyramidPool2D, self).__init__()

        self.pool  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNRelu(in_c, out_c, kernel_size=1),
        )

        self.conv1 = ConvBNRelu(in_c, out_c, kernel_size=1)
        self.conv2 = ConvBNRelu(in_c, out_c, padding=dilations[0], dilation=dilations[0])
        self.conv3 = ConvBNRelu(in_c, out_c, padding=dilations[1], dilation=dilations[1])
        self.conv4 = ConvBNRelu(in_c, out_c, padding=dilations[2], dilation=dilations[2])

        self.concat_conv = ConvBNRelu(out_c * 5, out_c, kernel_size=1)

    def forward(self, input_):
        pool  = self.pool(input_)
        # TODO : Remove up
        pool  = F.upsample(pool, size=input_.shape[2:], mode="bilinear")
        
        conv1 = self.conv1(input_)
        conv2 = self.conv2(input_)
        conv3 = self.conv3(input_)
        conv4 = self.conv4(input_)

        bridge = torch.cat([conv1, conv2, conv3, conv4, pool], dim=1)
        return self.concat_conv(bridge)