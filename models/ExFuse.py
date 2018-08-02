import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer.ExFuseLayer import SEB, CNA, GCN, ECRE, DAP, ExFuseLevel


class ExFuse(nn.Module):
    def __init__(self, resnet):
        super(ExFuse, self).__init__()
        self.conv1 = resnet.conv1
        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3
        self.res5 = resnet.layer4

        self.gcn5 = GCN(2048, 21)
        self.ecre = ECRE(21)

        self.level3 = ExFuseLevel(1024)
        self.level2 = ExFuseLevel(512)
        self.level1 = ExFuseLevel(256)

        self.dap = DAP(21)
        self.out = nn.Conv2d(21, 1, 1)

    def forward(self, input_):
        if input_.shape[1] == 1:
            input_ = torch.cat([input_, input_, input_], dim=1)
        print(input_.shape)
        x = self.conv1(input_)
        level1 = self.res2(x)
        level2 = self.res3(level1)
        level3 = self.res4(level2)
        level4 = self.res5(level3)

        l4_feature = self.gcn5(level4)
        l4_feature = self.ecre(l4_feature)

        l3_feature = self.level3(level3, level4, l4_feature)
        l2_feature = self.level2(level2, level3, l3_feature)
        l1_feature = self.level1(level1, level2, l2_feature)

        dap = self.dap(l1_feature)
        return self.out(dap)


if __name__ == "__main__":
    a = torch.randn([1, 1, 448, 448])
    from Resnet import resnet50
    resnet = resnet50()
    model = ExFuse(resnet)
    output = model(a)
    print("Done, ", output.shape)
