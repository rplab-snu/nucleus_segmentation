import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.unet_layer import UnetConv2D, UnetUpConv2D, UnetConv3D, UnetUpConv3D, weights_init_kaiming
from models.layers.dsv_layer import UnetUpConv2D_CT, UnetUpConv3D_CT, UnetDsv2D, UnetDsv3D
from models.layers.multi_attention_layer import MultiAttentionBlock2D, MultiAttentionBlock3D
from models.layers.grid_attention_layer import UnetGridGatingSignal2D, UnetGridGatingSignal3D

class unet_CT_multi_att_dsv_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(unet_CT_multi_att_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1    = UnetConv3D(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2    = UnetConv3D(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3    = UnetConv3D(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4    = UnetConv3D(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding=1)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3D(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding=1)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUpConv3D_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUpConv3D_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUpConv3D_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUpConv3D_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm3d):
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
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4           = self.up_concat4(g_conv4, center)

        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3           = self.up_concat3(g_conv3, up4)

        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2           = self.up_concat2(g_conv2, up3)
        up1           = self.up_concat1(conv1,   up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


class Unet_CT_multi_attention_dsv_2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=1,
                 nonlocal_mode='concatenation', attention_dsample=2, is_batchnorm=True):
        super(Unet_CT_multi_attention_dsv_2D, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1    = UnetConv2D(in_channels, filters[0], is_batchnorm, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2    = UnetConv2D(filters[0], filters[1], is_batchnorm, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3    = UnetConv2D(filters[1], filters[2], is_batchnorm, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4    = UnetConv2D(filters[2], filters[3], is_batchnorm, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # bridge
        self.center = UnetConv2D(filters[3], filters[4], is_batchnorm, kernel_size=3, padding=1)
        self.gating = UnetGridGatingSignal2D(filters[4], filters[4], kernel_size=1, is_batchnorm=is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock2D(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                     nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock2D(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                     nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock2D(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                     nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUpConv2D_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUpConv2D_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUpConv2D_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUpConv2D_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv2D(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv2D(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv2D(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv2d(n_classes*4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm3d):
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
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4           = self.up_concat4(g_conv4, center)

        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3           = self.up_concat3(g_conv3, up4)

        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2           = self.up_concat2(g_conv2, up3)
        up1           = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return final


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


if __name__ == "__main__":    
    input2D = torch.Tensor([1, 1, 448, 448])
    model = Unet_CT_multi_attention_dsv_2D()
    output2D = model(input2D)
    
    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)
