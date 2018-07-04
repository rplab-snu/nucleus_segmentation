import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channel=1, ndf=64):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel

        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),

            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

        self.fc_layer = nn.Linear(25 * 25, 1)
        self.out = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        out = self.conv_layer(input)
        out = self.fc_layer(out.view(-1))
        out = self.out(out)
        return out
