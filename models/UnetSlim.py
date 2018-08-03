import torch
import torch.nn as nn
from models.layers.unet_layer import weights_init_kaiming, UnetUpConv2D
from models.layers.ExFuseLayer import CNA, UpCNA
import torch.nn.functional as F


class UnetUpConvSlim(nn.Module):
    def __init__(self, in_c, out_c, norm=nn.InstanceNorm2d, is_deconv=False):
        super(UnetUpConvSlim, self).__init__()
        self.conv = CNA(in_c, out_c, norm=norm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        else:
            self.up = nn.UpsamplingBilinear2d(2)
           
        for m in self.children():
            if m.__class__.__name__.find("Conv") != -1:
                m.apply(weights_init_kaiming)

    def forward(self, input1, input2):
        output2 = self.up(input2)
        offset = output2.size()[2] - input1.size()[2]
        padding = [offset // 2] * 4
        output1 = F.pad(input1, padding)
        output = torch.cat([output1, output2], dim=1)
        return self.conv(output)

class UnetSlim(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1,
                 is_deconv=True, norm=nn.BatchNorm2d, is_pool=True):
        super(UnetSlim, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [x // feature_scale for x in filters]
        print("Unet Slim filters : ", filters)

        # downsampling
        self.conv1 = CNA(1, filters[0], norm=norm) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[0], filters[0], norm, stride=2)

        self.conv2 = CNA(filters[0], filters[1], norm=norm) 
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[1], filters[1], norm, stride=2)

        self.conv3 = CNA(filters[1], filters[2], norm=norm) 
        self.maxpool3 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[2], filters[2], norm, stride=2)

        self.conv4 = CNA(filters[2], filters[3], norm=norm) 
        self.maxpool4 = nn.MaxPool2d(kernel_size=2) if is_pool else ConvBNReLU(filters[3], filters[3], norm, stride=2)

        self.center = CNA(filters[3], filters[4], norm=norm)

        # upsampling
        self.up_concat4 = UnetUpConvSlim(filters[4], filters[3], norm, is_deconv)
        self.up_concat3 = UnetUpConvSlim(filters[3], filters[2], norm, is_deconv)
        self.up_concat2 = UnetUpConvSlim(filters[2], filters[1], norm, is_deconv)
        self.up_concat1 = UnetUpConvSlim(filters[1], filters[0], norm, is_deconv)

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
    model = UnetSlim()
    from torchsummary import summary
    summary(model.cuda(), (1,448,448))
    import torch
    input2D = torch.randn([1, 1, 448, 448])
    output2D = model(input2D)

    print("input shape : \t", input2D.shape)
    print("output shape  : \t", output2D.shape)
