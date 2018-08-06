import torch
import torch.nn as nn
from models.layers.ExFuseLayer import CNA, UpCNA, GCN, SEB, ECRE, DAP


class ExFuse(nn.Module):
    def __init__(self, resnet):
        super(ExFuse, self).__init__()
        # self.conv1 = resnet.conv1
        self.conv1 = CNA(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = resnet.layer1
        self.seb2 = SEB(256, 512)
        self.gcn2 = GCN(256, 21)
        self.upc2 = UpCNA(21, 21)

        self.res3 = resnet.layer2
        self.seb3 = SEB(512, 1024)
        self.gcn3 = GCN(512, 21)
        self.upc3 = UpCNA(21, 21)

        self.res4 = resnet.layer3
        self.seb4 = SEB(1024, 2048)
        self.gcn4 = GCN(1024, 21)
        self.upc4 = UpCNA(21, 21)

        self.res5 = resnet.layer4
        self.gcn5 = GCN(2048, 21)
        self.ecre = ECRE(21)

        self.out = nn.Sequential(UpCNA(21, 21),
                                 DAP(21),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(21, 1, 1))

    def forward(self, input_):
        x = self.conv1(input_)
        x = self.pool1(x)
        res2 = self.res2(x)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)

        level5 = self.gcn5(res5)
        level5 = self.ecre(level5)

        level4 = self.seb4(res4, res5)
        level4 = self.gcn4(level4)
        level4 += level5
        level4 = self.upc4(level4)

        level3 = self.seb3(res3, res4)
        level3 = self.gcn3(level3)
        level3 += level4
        level3 = self.upc3(level3)

        level2 = self.seb2(res2, res3)
        level2 = self.gcn2(level2)
        level2 += level3
        level2 = self.upc2(level2)

        out = self.out(level2)
        return out, level5
