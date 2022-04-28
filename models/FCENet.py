import torch.nn as nn
import torch
from torchsummary import summary
from models.resnet18 import Resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"

class Conv_block(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, kernel_size=3, stride=1, padding=1, init_weight=True):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        if init_weight is True:
            self._initialize_weight()

    # @torchsnooper.snoop()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def _initialize_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class Up_sampling(nn.Module):
    def __init__(self, in_channel=1, factor=2, init_weight=True):
        super(Up_sampling, self).__init__()
        out_channel = in_channel * factor
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0) # change the channels
        self.up = nn.PixelShuffle(factor)
        if init_weight is True:
            self._initialize_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

    def _initialize_weight(self):
        nn.init.xavier_normal_(self.conv.weight, gain=1.)



class Attention_Block(nn.Module):
    def __init__(self, init_weight=True):
        super(Attention_Block, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        if init_weight is True:
            self._initialize_weight()

    def forward(self, x):
        feature = x
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat((avg, max), dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(feature, x)
        return x

    def _initialize_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AttentionPath(nn.Module):
    def __init__(self, in_channel=1, out_channel=512, n_filter=32, init_weight=True):
        super(AttentionPath, self).__init__()
        self.resnet18 = Resnet18(in_channel=in_channel, out_channel=out_channel)
        self.atten = Attention_Block(init_weight=init_weight)
        self.conv = Conv_block(in_channel=128, out_channel=2 * n_filter, kernel_size=3, stride=1, padding=1, init_weight=init_weight)
        self.up1 = Up_sampling(in_channel=out_channel, factor=2, init_weight=init_weight)
        self.up2 = Up_sampling(in_channel=out_channel // 2, factor=2, init_weight=init_weight)

    def forward(self, x):
        _, feat16, feat32 = self.resnet18(x)
        feat32 = self.atten(feat32)
        feat32 = self.up1(feat32)

        feat16 = self.atten(feat16) + feat32
        # feat16 = self.atten(feat16)
        feat16 = self.up2(feat16)
        out = self.conv(feat16)
        return out

class SpatialPath(nn.Module):
    def __init__(self, in_channel=1, n_filter=32, init_weight=True):
        super(SpatialPath, self).__init__()
        self.conv1 = Conv_block(in_channel=in_channel, out_channel=n_filter, kernel_size=7, stride=2, padding=3, init_weight=init_weight)
        self.conv2 = Conv_block(in_channel=n_filter, out_channel=2 * n_filter, kernel_size=3, stride=2, padding=1, init_weight=init_weight)
        self.conv3 = Conv_block(in_channel=2 * n_filter, out_channel=2 * n_filter, kernel_size=3, stride=2, padding=1, init_weight=init_weight)

    def forward(self, x):
        x = self.conv1(x)
        out1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        return out1, x

class FCENet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, n_filter=32, init_weight=True):
        super(FCENet, self).__init__()
        self.sp = SpatialPath(in_channel=in_channel, n_filter=n_filter, init_weight=init_weight)
        self.tp = AttentionPath(in_channel=in_channel, n_filter=n_filter, init_weight=init_weight)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        # self.up1 = nn.ConvTranspose2d(4 * n_filter, 4 * n_filter, kernel_size=4, stride=4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.up2 = nn.ConvTranspose2d(2 * n_filter, 2 * n_filter, kernel_size=2, stride=2)
        self.conv1 = Conv_block(in_channel=4 * n_filter, out_channel=n_filter, kernel_size=3, stride=1, padding=1, init_weight=init_weight)
        self.conv2 = Conv_block(in_channel=2 * n_filter, out_channel=n_filter, kernel_size=3, stride=1, padding=1, init_weight=init_weight)
        self.conv1x1 = nn.Conv2d(n_filter, out_channel, kernel_size=1, stride=1, padding=0)
        if init_weight is True:
            self._initialize_weight()

    def forward(self, x):
        out1, sp = self.sp(x)
        tp = self.tp(x)
        x = torch.cat((sp, tp), dim=1)
        x = self.up1(x)
        x = self.conv1(x)
        x = torch.cat((out1, x), dim=1)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.conv1x1(x)
        return x

    def _initialize_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight, gain=1.)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ ==  "__main__":
    model = FCENet(in_channel=1, out_channel=1, n_filter=32).cuda()
    summary(model, input_size=(1, 512, 512))