import torch
import torch.nn as nn
from models.layers.resnet_layer import ResBlock, ResBlockMultiGrid
from models.layers.AtrousSpatialPyramidPool import ConvBNReLU, AtrousSpatilPyramidPool2D

class DeeplabV3P(nn.Module):
    def __init__(self, n_blocks=(3, 4, 23, 3), n_classes=1, dilation=(6, 12, 18), multi_grid=(1, 2, 1)):
        super(DeeplabV3P, self).__init__()
        self.layer1 = nn.Sequential(
            ConvBNReLU(1, 64, 7, 2, 3, 1),
            nn.Maxpool2d(3, 2, 1, ceil_mode=True)
        )
        #                      n_layer     in_c  mid_c out_c stride dilation
        self.layer2 = ResBlock(n_blocks[0], 64,  64,   256,  1, 1)) # output_stride=4
        self.layer3 = ResBlock(n_blocks[1], 256, 128,  512,  2, 1)) # output_stride=8
        self.layer4 = ResBlock(n_blocks[2], 512, 256,  1024, 1, 2)) # output_stride=8

        self.layer5 = ResBlockMG(n_blocks[3], 1024, 512, 2048, 1, 2, mg=multi_grid))

        self.aspp = AtrousSpatilPyramidPool2D(256, 256, dilation))

        self.fc = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1))
        )

    def forward(self, input_):
        layer1 = self.layer1(input_)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)

        aspp = self.aspp(layer5)
        return self.fc(aspp)


if __name__ == "__model__":
    input_ = torch.Tensor([1, 1, 448, 448])
    model  = DeeplabV3P()
    output = model(input_)
    print("output shape : ", output.shape)