import torch.nn as nn
from models.layers.UnetResLayer import UnetResConv2D, UnetResUpConv2D, weights_init_kaiming, SpatialBridge
import torch.nn.functional as F

class UnetRes2D(nn.Module):

    def __init__(self, n_classes, norm):
        super(UnetRes2D, self).__init__()
        print("UnetRes2D Pool")
        filters = [64, 128]

        # downsampling
        self.conv1    = UnetResConv2D(1, filters[0], 8, norm)
        self.maxpool1 = nn.Sequential(nn.Conv2d(filters[0], filters[0], 3, 2, 1),
                                      norm(filters[0]),
                                      nn.ReLU(True))

        self.conv2    = UnetResConv2D(filters[0], filters[1], 16, norm)
        self.maxpool2 = nn.Sequential(nn.Conv2d(filters[1], filters[1], 3, 2, 1), 
                                      norm(filters[1]),
                                      nn.ReLU(True))

        dilates = [1, 2, 5]
        groups  = [32, 64, 128]
        self.center   = SpatialBridge(filters[1], dilates, groups, norm)

        # upsampling
        self.up_concat2 = UnetResUpConv2D(filters[1], filters[0], 8, norm)
        self.up_concat1 = UnetResUpConv2D(filters[0], n_classes, 8, norm)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.BatchNorm2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.InstanceNorm2d):
                m.apply(weights_init_kaiming)
            elif isinstance(m, nn.GroupNorm2d):
                m.apply(weights_init_kaiming)

    def forward(self, inputs):
        conv1    = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2    = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        center = self.center(maxpool4)

        up2 = self.up_concat2(conv2, center)
        up1 = self.up_concat1(conv1, up2)

        return up1 

if __name__ == "__main__":
    import torch
    input2D = torch.randn([1, 1, 448, 448])
    print("input shape : \t", input2D.shape)
    model = UnetSH2D(3)
    output2D = model(input2D)
    print("output shape  : \t", output2D.shape)
