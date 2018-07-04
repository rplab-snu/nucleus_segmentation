import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.unet_layer import UnetConv2D, UnetUpConv2D, UnetConv3D, UnetUpConv3D, weights_init_kaiming
from models.layers.grid_attention_layer import UnetGridGatingSignal2D, UnetGridGatingSignal3D, GridAttentionBlock2D,GridAttentionBlock3D 


class Unet_GridAttention2D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=1,
                 nonlocal_mode='concatenation', attention_dsample=2, is_batchnorm=True):

        super(Unet_GridAttention2D, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1    = UnetConv2D(1, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2    = UnetConv2D(filters[0],  filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3    = UnetConv2D(filters[1],  filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4    = UnetConv2D(filters[2],  filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center   = UnetConv2D(filters[3],  filters[4], is_batchnorm)
        self.gating = UnetGridGatingSignal2D(filters[4], filters[3], kernel_size=1, is_batchnorm=is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1],    gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=2, mode=nonlocal_mode)

        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2],    gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=2, mode=nonlocal_mode)

        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3],    gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=2, mode=nonlocal_mode)
                                                    
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
        # Feature Extraction
        conv1    = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2    = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3    = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4    = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # att* 를 visualize 해볼까?
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        g_conv3, att3 = self.attentionblock3(conv3, gating)
        g_conv2, att2 = self.attentionblock2(conv2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up_concat4(g_conv4, center)
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


class Unet_GridAttention3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(Unet_GridAttention3D, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1     = UnetConv3D(in_channels, filters[0], is_batchnorm)
        self.maxpool1  = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2     = UnetConv3D(filters[0],  filters[1], is_batchnorm)
        self.maxpool2  = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3     = UnetConv3D(filters[1],  filters[2], is_batchnorm)
        self.maxpool3  = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4    = UnetConv3D(filters[2],  filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3D(filters[3], filters[4], is_batchnorm)
        self.gating = UnetGridGatingSignal3D(filters[4], filters[3], kernel_size=(1, 1, 1), is_batchnorm=is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock3D(in_channels=filters[1], gating_channels=filters[3],
                                                    inter_channels=filters[1], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock3 = GridAttentionBlock3D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2], sub_sample_factor=attention_dsample, mode=nonlocal_mode)
        self.attentionblock4 = GridAttentionBlock3D(in_channels=filters[3], gating_channels=filters[3],
                                                    inter_channels=filters[3], sub_sample_factor=attention_dsample, mode=nonlocal_mode)

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
        # Feature Extraction
        conv1    = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2    = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3    = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4    = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # att* 를 visualize 해볼까?
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        g_conv3, att3 = self.attentionblock3(conv3, gating)
        g_conv2, att2 = self.attentionblock2(conv2, gating)

        # Upscaling Part (Decoder)
        up4 = self.up_concat4(g_conv4, center)
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


if __name__ == "__main__":
    input2D = torch.Tensor([1, 1, 448, 448])
    model = Unet_GridAttention2D()
    output2D = model(input2D)
    
    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)




