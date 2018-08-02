import torch
import torch.nn as nn
from models.layers.unet_layer import UnetConv2D, UnetUpConv2D, weights_init_kaiming, ConvBNReLU
from models.layers.ExFuseLayer import SEB, CNA, GCN, ECRE, DAP, UnetExFuseLevel


class UnetGCN(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, norm=nn.InstanceNorm2d, is_pool=True):
        super(UnetGCN, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[0], filters[0], norm, stride=2)

        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[1], filters[1], norm, stride=2)

        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[2], filters[2], norm, stride=2)

        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[3], filters[3], norm, stride=2)

        self.center = UnetConv2D(filters[3], filters[4], norm)

        # upsampling
        self.up_concat4 = UnetUpConv2D(filters[4], filters[3], norm, is_deconv)
        self.up_concat3 = UnetUpConv2D(filters[3], filters[2], norm, is_deconv)
        self.up_concat2 = UnetUpConv2D(filters[2], filters[1], norm, is_deconv)
        self.up_concat1 = UnetUpConv2D(filters[1], filters[0], norm, is_deconv)

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
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


class UnetGCNSEB(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, norm=nn.InstanceNorm2d, is_pool=True):
        super(UnetGCNSEB, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[0], filters[0], norm, stride=2)

        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[1], filters[1], norm, stride=2)

        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[2], filters[2], norm, stride=2)

        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[3], filters[3], norm, stride=2)
        self.center = UnetConv2D(filters[3], filters[4], norm)

        # upsampling
        self.up_concat4 = SEB(filters[4], filters[3])
        self.up_concat3 = SEB(filters[3], filters[2])
        self.up_concat2 = SEB(filters[2], filters[1])
        self.up_concat1 = SEB(filters[1], filters[0])

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
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


class UnetUpECRE(nn.Module):
    def __init__(self, in_size, out_size, norm, is_deconv=False):
        super(UnetUpECRE, self).__init__()

        self.conv = UnetConv2D(in_size + out_size, out_size, norm)
        self.up = ECRE(in_size)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2D') != -1:
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset = output2.size()[2] - input1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(input1, padding)
        output = torch.cat([output1, output2], 1)
        return self.conv(output)


class UnetGCNECRE(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, norm=nn.InstanceNorm2d, is_pool=True):
        super(UnetGCNECRE, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[0], filters[0], norm, stride=2)

        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[1], filters[1], norm, stride=2)

        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[2], filters[2], norm, stride=2)

        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[3], filters[3], norm, stride=2)
        self.center = UnetConv2D(filters[3], filters[4], norm)

        # upsampling
        self.up_concat4 = UnetUpECRE(filters[4], filters[3], norm)
        self.up_concat3 = UnetUpECRE(filters[3], filters[2], norm)
        self.up_concat2 = UnetUpECRE(filters[2], filters[1], norm)
        self.up_concat1 = UnetUpECRE(filters[1], filters[0], norm)

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
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        return final


class UnetExFuse(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, norm=nn.InstanceNorm2d, is_pool=True):
        super(UnetExFuse, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]

        # downsampling
        self.conv1 = UnetConv2D(1, filters[0], norm)
        self.gcn1 = GCN(filters[0], filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[0], filters[0], norm, stride=2)

        self.conv2 = UnetConv2D(filters[0], filters[1], norm)
        self.gcn2 = GCN(filters[1], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[1], filters[1], norm, stride=2)

        self.conv3 = UnetConv2D(filters[1], filters[2], norm)
        self.gcn3 = GCN(filters[2], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[2], filters[2], norm, stride=2)

        self.conv4 = UnetConv2D(filters[2], filters[3], norm)
        self.gcn4 = GCN(filters[3], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[3], filters[3], norm, stride=2)

        self.center = UnetConv2D(filters[3], filters[4], norm)

        # upsampling
        self.up_concat4 = UnetUpConv2D(filters[4], filters[3], norm, is_deconv)
        self.level4 = UnetExFuseLevel(filters[3], filters[2])
        self.level3 = UnetExFuseLevel(filters[2], filters[1])
        self.level2 = UnetExFuseLevel(filters[1], filters[0])
        self.final = nn.Sequential(DAP(filters[0]), nn.Conv2d(filters[0], 1, 1))
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = self.gcn1(conv1)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        conv2 = self.gcn2(conv2)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        conv3 = self.gcn3(conv3)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        conv4 = self.gcn4(conv4)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        l4 = self.level4(conv4, center, up4)
        l3 = self.level3(conv3, conv4, l4)
        l2 = self.level2(conv2, conv3, l3)
        final = self.final(l2)
        return final
