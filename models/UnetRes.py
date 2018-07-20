import torch.nn as nn
from models.layers.UnetResLayer import UnetResConv2D, UnetResUpConv2D, weights_init_kaiming, ConvNormReLU
import torch.nn.functional as F


class UnetRes2D(nn.Module):

    def __init__(self, n_classes, norm, is_pool=False):
        super(UnetRes2D, self).__init__()
        print("UnetRes2D")
        ch = [16, 32, 64, 128, 256]

        # downsampling
        self.conv1    = UnetResConv2D(1, ch[0], norm, ch[0] / 8)
        self.maxpool1 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[0], ch[0], norm, stride=2)

        self.conv2    = UnetResConv2D(ch[0], ch[1], norm, ch[1] / 8)
        self.maxpool2 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[1], ch[1], norm, stride=2)

        self.conv3    = UnetResConv2D(ch[1], ch[2], norm, ch[2] / 8)
        self.maxpool3 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[2], ch[2], norm, stride=2)

        self.conv4    = UnetResConv2D(ch[2], ch[3], norm, ch[3] / 8)
        self.maxpool4 = nn.MaxPool2d(2) if is_pool else ConvNormReLU(ch[3], ch[3], norm, stride=2)

        self.center = UnetResConv2D(ch[3], ch[4], norm, ch[4] / 8)
        # upsampling
        #                                                     group size
        self.up_concat4 = UnetResUpConv2D(ch[4], ch[3], norm, ch[3] / 8)
        self.up_concat3 = UnetResUpConv2D(ch[3], ch[2], norm, ch[2] / 8)
        self.up_concat2 = UnetResUpConv2D(ch[2], ch[1], norm, ch[1] / 8)
        self.up_concat1 = UnetResUpConv2D(ch[1], ch[0], norm, ch[0] / 8)

        self.final = nn.Conv2d(ch[0], 1, 1, 1, 0) 

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.InstanceNorm2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.GroupNorm):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1    = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2    = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3    = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4    = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        return self.final(up1) 

if __name__ == "__main__":
    import torch
    device = torch.device("cuda")
    input2D = torch.randn([1, 1, 448, 448]).to(device)
    print("input shape : \t", input2D.shape)
    model = UnetRes2D(1, nn.InstanceNorm2d)
    output2D = model(input2D).to(device)
    print("output shape  : \t", output2D.shape)
    from torchsummary import summary
    summary(model, input_size=(1, 448, 448))
