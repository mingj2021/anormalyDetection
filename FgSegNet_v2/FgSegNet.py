import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import torch.nn.functional as F

"""
for param in model.parameters():
    param.requires_grad = False
"""


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        if self.act:
            self.act = nn.ReLU()

    def forward(self, x):
        if self.act:
            return self.act(self.conv(x))
        else:
            return self.conv(x)


# model = torchvision.models.vgg16()
# summary(model.features.to('cuda:0'), input_size=(3, 240, 320))
# print(model.features)
# print(model.avgpool)
# del model.classifier[-1]
# print(model.classifier)


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 64, 3)
        self.conv2 = Conv(64, 64, 3)
        self.m1 = nn.MaxPool2d(2, 2, 0)

        self.conv3 = Conv(64, 128, 3)
        self.conv4 = Conv(128, 128, 3)
        self.m2 = nn.MaxPool2d(2, 2, 0)

        self.conv5 = Conv(128, 256, 3)
        self.conv6 = Conv(256, 256, 3)
        self.conv7 = Conv(256, 256, 3)

        self.conv8 = Conv(256, 512, 3)
        self.dr1 = nn.Dropout2d(0.5)
        self.conv9 = Conv(512, 512, 3)
        self.dr2 = nn.Dropout2d(0.5)
        self.conv10 = Conv(512, 512, 3)
        self.dr3 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        a = x
        x = self.m1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        b = x
        x = self.m2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.conv8(x)
        x = self.dr1(x)
        x = self.conv9(x)
        x = self.dr2(x)
        x = self.conv10(x)
        x = self.dr3(x)

        return x, a, b


class M_FPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = nn.MaxPool2d(3, 1, 1)
        self.conv1 = Conv(512, 64, act=False)

        self.conv2 = Conv(512, 64, 3, act=False)

        self.act1 = nn.ReLU()
        self.conv3 = Conv(576, 64, 3, d=4, act=False)

        self.act2 = nn.ReLU()
        self.conv4 = Conv(576, 64, 3, d=8, act=False)

        self.act3 = nn.ReLU()
        self.conv5 = Conv(576, 64, 3, d=16, act=False)

        self.norm1 = nn.InstanceNorm2d(320)

        self.act4 = nn.ReLU()
        self.dr1 = nn.Dropout2d(0.25)

    def forward(self, x):
        pool = self.m1(x)
        pool = self.conv1(pool)

        d1 = self.conv2(x)

        y = torch.concatenate([x, d1],dim=1)
        y = self.act1(y)
        d4 = self.conv3(y)

        y = torch.concatenate([x, d4],dim=1)
        y = self.act2(y)
        d8 = self.conv4(y)

        y = torch.concatenate([x, d8],dim=1)
        y = self.act3(y)
        d16 = self.conv4(y)

        x = torch.concatenate([pool, d1, d4, d8, d16],dim=1)
        x = self.norm1(x)
        x = self.act4(x)
        x = self.dr1(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.AvgPool2d((240, 320))
        self.conv1 = Conv(128,64,act=False)
        self.p2 = nn.AvgPool2d((120, 160))

        self.conv2 = Conv(320,64,3,act=False)
        self.norm1 = nn.InstanceNorm2d(64)
        self.act1 = nn.ReLU()
        self.upsamp1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3 = Conv(64,64,3,act=False)
        self.norm2 = nn.InstanceNorm2d(64)
        self.act2 = nn.ReLU()
        self.upsamp2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv4 = Conv(64,64,3,act=False)
        self.norm3 = nn.InstanceNorm2d(64)
        self.act3 = nn.ReLU()

        self.conv5 = Conv(64,1,1,act=False)
        self.act4 = nn.Sigmoid()

    def forward(self, x,a,b):
        a = self.p1(a)
        b = self.conv1(b)
        b = self.p2(b)
        
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.act1(x)
        x1 = torch.multiply(x,b)
        x = torch.add(x,x1)
        x = self.upsamp1(x)

        x = self.conv3(x)
        x = self.norm2(x)
        x = self.act2(x)
        x2 = torch.multiply(x,a)
        x = torch.add(x,x2)
        x = self.upsamp2(x)

        x = self.conv4(x)
        x = self.norm3(x)
        x = self.act3(x)

        x = self.conv5(x)
        x = self.act4(x)
        return x


class FgSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = VGG16()
        self.model2 = M_FPM()
        self.model3 = Decoder()

    def forward(self,x):
        x, a, b = self.model1(x)
        x = self.model2(x)
        x = self.model3(x,a,b)
        return x

if __name__ == '__main__':
    input = torch.randn(1, 3, 240, 320).to("cuda:0")
    model = FgSegNet().to("cuda:0")
    x = model(input)
    summary(model, input_size=(3, 240, 320))
    # input = torch.randn(1, 3, 240, 320, requires_grad=True)
    # target = torch.rand(1, 1, 240, 320, requires_grad=False).to("cuda:0")
    # loss = F.binary_cross_entropy(x, target)
    # print(loss)
    # loss.backward()
