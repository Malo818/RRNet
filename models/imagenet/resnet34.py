from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, inchanel, outchanel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchanel, outchanel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchanel, outchanel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchanel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.layer1 = self.__make_layer__(3, 32, 3, stride=2)
        self.layer2 = self.__make_layer__(32, 64, 4, stride=2)
        self.layer3 = self.__make_layer__(64, 128, 6, stride=2)
        self.layer4 = self.__make_layer__(128, 256, 3, stride=2)

    def __make_layer__(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # x（b4，3，156，156）
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# model = ResNet34()
# summary(model, (156, 156, 3), device='cpu')