import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    '''
    expansion通道扩充比例
    out_channels就是输出的channel
    '''


def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()

    self.residual_function = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels * BasicBlock.expansion)
    )

    self.shortcut = nn.Sequential()

    if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )


def forward(self, x):
    return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    '''
    espansion是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    '''

    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)

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

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out
class Classification_only(nn.Module):
    def __init__(self,batch_size=1, num_classes=6, baseline=True, attention= True):
        super(Classification_only,self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        #self.SpatialAttention = SpatialAttention()
        self.maxpool = nn.AdaptiveAvgPool2d((1,1))
        #print('using MLT training')
        if baseline:
            print('-----using SI baseline-----')
            self.fc = nn.Linear(64,self.num_classes)
        else:
            print("-----using lowest-----")
            self.fc = nn.Linear(512,self.num_classes)
            #self.fc2 = nn.Linear(64,6)
        self.relu = nn.ReLU()
        self.mode = baseline
        self.using_att = attention
        if self.using_att:
            print('----applying saptial attention')
        self.sig = nn.Sigmoid()

        #self.fc2 = nn.Linear(batch_size,)
    def forward(self,x):
        batch_size = x.size(0)
        # if self.using_att:
        #     x_weight = self.SpatialAttention(x)
        #     x = x*x_weight

        x = self.maxpool(x)
        x=x.squeeze().view(batch_size,-1)
        x = self.fc(x)
        x = self.sig(x)
        return x

class UResnet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=1, attention=True):
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        print('using 2020MIA highest branch for training, encoder backbone is Resnet 50')
        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = SingleConv(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[1], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[2], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[3], layers[3], 1)
        #
        self.conv3_1 = VGGBlock(1024+2048, 1024, 1024)
        self.conv2_2 = VGGBlock(512+1024, 512, 512)
        self.conv1_3 = VGGBlock(256+512, 256, 256)
        self.conv0_4 = VGGBlock(64+256, 64, 64)

        self.final = nn.Conv2d(nb_filter[0], num_classes*3, kernel_size=1)
        self.attention  = attention
        self.classifer = Classification_only()
    def _make_layer(self, block, middle_channel, num_blocks, stride):
        '''
        middle_channels中间维度，实际输出channels = middle_channels * block.expansion
        num_blocks，一个Layer包含block的个数
        '''

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        mean = images.mean(dim=[1, 2, 3], keepdim=True)
        std = images.std(dim=[1, 2, 3], keepdim=True) + eps
        images -= mean
        images /= std
        # images = images.to(self.pixel_mean.device)
        # images = (images - self.pixel_mean) / self.pixel_std
        images = images.expand(-1, 3, -1, -1)
        return images

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        x_c = self.up(x0_4)
        output = self.final(x_c)
        heatmap = torch.sigmoid(output[:, :24, :, :])
        regression_x = output[:, 24:2 * 24, :, :]
        regression_y = output[:, 2 * 24:, :, :]
        class_output = self.classifer(x_c)
        return heatmap,regression_x,regression_y,class_output
