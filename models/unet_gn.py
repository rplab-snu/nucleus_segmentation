import torch.nn as nn
from models.layers.unet_gn_layer import UnetGNConv2D, UnetUpGNConv2D, weights_init_kaiming, ConvGNReLU
import torch.nn.functional as F


class UnetGN2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, is_pool=True):
        super(UnetGN2D, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1 = UnetGNConv2D(1, filters[0], filters[0] // 4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvGNReLU(filters[0], filters[0], filters[0]//4, stride=2)

        self.conv2 = UnetGNConv2D(filters[0], filters[1], filters[1] // 4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvGNReLU(filters[1], filters[1], filters[1]//4, stride=2)

        self.conv3 = UnetGNConv2D(filters[1], filters[2], filters[2] // 4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvGNReLU(filters[2], filters[2], filters[2]//4, stride=2)

        self.conv4 = UnetGNConv2D(filters[2], filters[3], filters[3] // 4)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvGNReLU(filters[3], filters[3], filters[3]//4, stride=2)

        self.center = UnetGNConv2D(filters[3], filters[4], filters[4] // 4)

        # upsampling
        self.up_concat4 = UnetUpGNConv2D(filters[4], filters[3], filters[3] // 4, is_deconv)
        self.up_concat3 = UnetUpGNConv2D(filters[3], filters[2], filters[2] // 4, is_deconv)
        self.up_concat2 = UnetUpGNConv2D(filters[2], filters[1], filters[1] // 4, is_deconv)
        self.up_concat1 = UnetUpGNConv2D(filters[1], filters[0], filters[0] // 4, is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


if __name__ == "__main__":
    import torch
    input2D = torch.randn([1, 1, 448, 448])
    model = Unet2D()
    output2D = model(input2D)

    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)

    from torchsummary import summary
    summary(model.cuda(), (1,448,448))
