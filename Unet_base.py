import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
                                   nn.BatchNorm2d(1, momentum=0.01, affine=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Classification_only(nn.Module):
    def __init__(self,batch_size=1, num_classes=5, baseline=True, using_attention= True):
        super(Classification_only,self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.maxpool = nn.AdaptiveAvgPool2d((1,1))
        #print('using MLT training')
        if baseline:
            print('-----using MLT UNet baseline-----')
            self.fc = nn.Linear(64,self.num_classes)
        else:
            print("-----using cUnet for training-----")
            self.fc = nn.Linear(512,self.num_classes)
        if using_attention:
            self.SpatialAttention = SpatialAttention()
            #self.fc2 = nn.Linear(64,6)
        self.relu = nn.ReLU()
        self.mode = baseline
        self.using_att = using_attention
        self.sig = nn.Sigmoid()

        #self.fc2 = nn.Linear(batch_size,)
    def forward(self,x):
        batch_size = x.size(0)
        if self.using_att:
            x_weight = self.SpatialAttention(x)
            x = x*x_weight

        x = self.maxpool(x)
        x=x.squeeze().view(batch_size,-1)
        x = self.fc(x)
        x = self.sig(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_base(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, bilinear=True,classification= True):
        super(UNet_base, self).__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.classifictaionbranch = classification

        for i, (n_chan, n_class) in enumerate(zip(in_channels, out_channels)):
            setattr(self, 'in{i}'.format(i=i), OutConv(n_chan, 64))
            setattr(self, 'out{i}'.format(i=i), OutConv(64, n_class*3))
        self.conv = DoubleConv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        if classification:
            print('using Unet baseline')
            self.classifier_only = Classification_only(baseline=True,using_attention=True)
        else:
            print('using cUnet for training')
            self.classifier_only = Classification_only(baseline=False,using_attention=True)
    def forward(self, x):
        x1 = getattr(self, 'in{}'.format(0))(x)
        x1 = self.conv(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = getattr(self, 'out{}'.format(0))(x)
        heatmap = torch.sigmoid(logits[:, :20, :, :])
        # heatmap = self.softmax(logits[:,:24,:,:])
        regression_x = logits[:, 20:2 * 20, :, :]
        regression_y = logits[:, 2 * 20:, :, :]

        if self.classifictaionbranch ==True:
            ##only classification branch
                #print("using MLT branch")
            class_output = x
            class_output = self.classifier_only(class_output)
            return heatmap, regression_x, regression_y, class_output
        else:
            class_output = x5
            class_output = self.classifier_only(class_output)
            return heatmap, regression_x, regression_y, class_output