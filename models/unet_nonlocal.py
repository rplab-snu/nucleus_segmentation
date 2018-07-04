import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.unet_layer import UnetConv2D, UnetUpConv2D, UnetConv3D, UnetUpConv3D, weights_init_kaiming
from models.layers.nonlocal_layer import NONLocalBlock2D, NONLocalBlock3D

class Unet_NonLocal2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='embedded_gaussian', nonlocal_sf=4):

        super(Unet_NonLocal2D, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1     = UnetConv2D(1, filters[0], is_batchnorm)
        self.maxpool1  = nn.MaxPool2d(kernel_size=2)
        self.nonlocal1 = NONLocalBlock2D(in_channels=filters[0], inter_channels=filters[0] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv2     = UnetConv2D(filters[0],  filters[1], is_batchnorm)
        self.maxpool2  = nn.MaxPool2d(kernel_size=2)
        self.nonlocal2 = NONLocalBlock2D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv3    = UnetConv2D(filters[1],  filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4    = UnetConv2D(filters[2],  filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center   = UnetConv2D(filters[3],  filters[4], is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUpConv2D(filters[4], filters[3], is_deconv)
        self.up_concat3 = UnetUpConv2D(filters[3], filters[2], is_deconv)
        self.up_concat2 = UnetUpConv2D(filters[2], filters[1], is_deconv)
        self.up_concat1 = UnetUpConv2D(filters[1], filters[0], is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)


    def forward(self, inputs):
        conv1     = self.conv1(inputs)
        maxpool1  = self.maxpool1(conv1)
        nonlocal1 = self.nonlocal1(maxpool1)

        conv2     = self.conv2(nonlocal1)
        maxpool2  = self.maxpool2(conv2)
        nonlocal2 = self.nonlocal2(maxpool2)

        conv3    = self.conv3(nonlocal2)
        maxpool3 = self.maxpool3(conv3)

        conv4    = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


class Unet_NonLocal3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(Unet_NonLocal3D, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1     = UnetConv3D(in_channels, filters[0], is_batchnorm)
        self.maxpool1  = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2     = UnetConv3D(filters[0],  filters[1], is_batchnorm)
        self.nonlocal2 = NONLocalBlock3D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)
        self.maxpool2  = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3     = UnetConv3D(filters[1],  filters[2], is_batchnorm)
        self.nonlocal3 = NONLocalBlock3D(in_channels=filters[2], inter_channels=filters[2] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)
        self.maxpool3  = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4    = UnetConv3D(filters[2],  filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3D(filters[3], filters[4], is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUpConv3D(filters[4], filters[3], is_deconv, is_batchnorm)
        self.up_concat3 = UnetUpConv3D(filters[3], filters[2], is_deconv, is_batchnorm)
        self.up_concat2 = UnetUpConv3D(filters[2], filters[1], is_deconv, is_batchnorm)
        self.up_concat1 = UnetUpConv3D(filters[1], filters[0], is_deconv, is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)


    def forward(self, inputs):
        conv1     = self.conv1(inputs)
        maxpool1  = self.maxpool1(conv1)

        conv2     = self.conv2(maxpool1)
        nonlocal2 = self.nonlocal2(conv2)
        maxpool2  = self.maxpool2(nonlocal2)

        conv3     = self.conv3(maxpool2)
        nonlocal3 = self.nonlocal3(conv3)
        maxpool3  = self.maxpool3(nonlocal3)

        conv4     = self.conv4(maxpool3)
        maxpool4  = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4,     center)
        up3 = self.up_concat3(nonlocal3, up4)
        up2 = self.up_concat2(nonlocal2, up3)
        up1 = self.up_concat1(conv1,     up2)

        final = self.final(up1)
        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


if __name__ == "__main__":
    input2D = torch.Tensor([1, 1, 448, 448])
    model = Unet_NonLocal2D()
    output2D = model(input2D)
    
    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)




