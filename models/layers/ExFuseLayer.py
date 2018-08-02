import torch
import torch.nn as nn
import torch.nn.functional as F


# Conv-Norm-Activation
class CNA(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1,
                 norm=nn.InstanceNorm2d, act=nn.ReLU):
        super(CNA, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
                                   norm(out_c), act(True))

    def forward(self, x):
        return self.layer(x)


# Semantic Embedding Branch, Fig 4
class SEB(nn.Module):
    def __init__(self, high_feature_c, low_feature_c,
                 up_scale=2):
        super(SEB, self).__init__()
        self.conv = CNA(high_feature_c, low_feature_c)
        self.up = nn.UpsamplingBilinear2d(scale_factor=up_scale)
        self.up_mode = up_mode
        self.up_scale = up_scale

    def forward(self, low_feature, high_feature):
        high_feature = self.conv(high_feature)
        high_feature = self.up(high_feature)
        return low_feature * high_feature # element wise mul


# Global Convolution Network
# https://github.com/ycszen/pytorch-segmentation
# https://arxiv.org/pdf/1703.02719.pdf
class GCN(nn.Module):
    def __init__(self, in_c, out_c, ks=7):
        super(GCN, self).__init__()
        self.conv_l1 = CNA(in_c, out_c, kernel_size=(ks, 1),
                           padding=(ks // 2, 0))
        self.conv_l2 = CNA(out_c, out_c, kernel_size=(1, ks),
                           padding=(0, ks // 2))

        self.conv_r1 = CNA(in_c, out_c, kernel_size=(1, ks),
                           padding=(0, ks // 2))
        self.conv_r2 = CNA(out_c, out_c, kernel_size=(ks, 1),
                           padding=(ks // 2, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        return x_l + x_r


# Explicit Channel Resolution Embedding
class ECRE(nn.Module):
    def __init__(self, in_c, up_scale=2):
        super(ECRE, self).__init__()
        self.ecre = nn.Sequential(CNA(in_c, in_c * up_scale * up_scale),
                                  nn.PixelShuffle(up_scale))

    def forward(self, input_):
        return self.ecre(input_)


# Semantic Supervision
class SS(nn.Module):
    # Fig 3
    def __init__(self, in_c, hidden_c, out_c):
        super(SS, self).__init__()
        # TODO : Channel Check
        self.conv = nn.Sequential(CNA(in_c, hidden_c), CNA(hidden_c, out_c))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(out_c, out_C),
                                nn.ReLU(True),
                                nn.Linear(out_c, out_c))

    def forward(self, input_):
        x = self.conv(input_)
        x = self.pool(x).view(input_shape[0], -1)
        return self.fc(x)


# Densely Adjacent Prediction
class DAP(nn.Module):
    def __init__(self, in_c, k=3):
        super(DAP, self).__init__()
        self.k2 = k * k
        self.conv = CNA(in_c, in_c * k * k)
        self.padd = nn.ZeroPad2d(k // 2)

    def forward(self, input_):
        batch, input_c, max_i, max_j = input_.shape
        x = self.conv(input_)
        x = self.padd(x)

        # It works only k=3.
        # TODO : Make beautiful
        a = [(0, max_i,     0, max_j), (0, max_i,     1, max_j + 1), (0, max_i,     2, max_j + 2),
             (1, max_i + 1, 0, max_j), (1, max_i + 1, 1, max_j + 1), (1, max_i + 1, 2, max_j + 2),
             (2, max_i + 2, 0, max_j), (2, max_i + 2, 1, max_j + 1), (2, max_i + 2, 2, max_j + 2)]

        R = torch.zeros([batch, input_c, self.k2, max_i * max_j]).cuda()
        for dap_c in range(input_c):
            for c, (s_i, e_i, s_j, e_j) in enumerate(a):
                R[:, dap_c, c] = x[:, c, s_i:e_i, s_j:e_j].contiguous().view(batch, -1)

        R = torch.mean(R, 2).reshape(batch, input_c, max_i, max_j)
        return R


# Each Step of Framework
class ExFuseLevel(nn.Module):
    def __init__(self, in_c, out_c=21, norm=nn.InstanceNorm2d):
        super(ExFuseLevel, self).__init__()
        self.seb = SEB(in_c * 2, in_c)
        self.gcn = GCN(in_c, out_c)
        self.upconv = nn.Sequential(nn.ConvTranspose2d(out_c, out_c, 3, 2, 1, output_padding=1),
                                    norm(out_c),
                                    nn.ReLU(True))

    def forward(self, low_level, high_level, prev_feature):
        level = self.seb(low_level, high_level)
        level = self.gcn(level)

        return self.upconv(level + prev_feature)

