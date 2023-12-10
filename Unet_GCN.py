import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from torch.autograd import Variable
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import global_mean_pool


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



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Classification_sim(nn.Module):
    def __init__(self,batch_size=1, num_classes=1):
        super(Classification_sim,self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        print('using AGL for training')
        self.avgpool2d = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool1d = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.ModuleList(nn.Linear(64,1)for i in range(6))
        #self.fc1 =nn.Linear(64,64)
        #self.dynamic = nn.Linear(64,128)
        self.relu = nn.ReLU(inplace=True)
        #self.fc = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()

        #self.fc2 = nn.Linear(batch_size,)
    def forward(self,x):
        x_all = []
        if len(x.size())== 4:
            x = self.avgpool2d(x)
        for i in range(x.size(1)):
            #x_sim = x[:, i]
        #     # #anmatual attention resnet
        #     if i< x.size(1)-2:
        #         x_sim= x[:,i]
        #         # parameters = self.dynamic(x)
        #         # param1 = parameters[:, :, :64].contiguous().view(-1, 64, 24)
        #         # param2 = parameters[:, :, 64:].contiguous().view(-1, 24, 64)
        #         # features = torch.bmm(x, param1)
        #         # features = torch.bmm(features, param2)
        #         # x_nearest = features[:,i+2]
        #         #print(x_sim.size())
        #         #print(x_nearest.size())
        #         x_nearest = x[:, i + 2]
        #         x_diff = x_sim-x_nearest
        #         x_sim = x_sim+x_diff
        #     else:
        #         x_sim =x[:,i]
        #         # parameters = self.dynamic(x)
        #         # param1 = parameters[:, :, :64].contiguous().view(-1, 64, 24)
        #         # param2 = parameters[:, :, 64:].contiguous().view(-1, 24, 64)
        #         # features = torch.bmm(x, param1)
        #         # features = torch.bmm(features, param2)
        #         # x_nearest = features[:,i-2]
        #         x_nearest = x[:, i - 2]
        #         x_diff = x_sim - x_nearest
        #         x_sim = x_sim+x_diff
            #x_sim = self.relu(self.fc1(x_sim))

            x_sim = x[:, i]
            x_sim=x_sim.squeeze().view(-1,64)
            x_sim = self.fc[i](x_sim)
            x_all.append(x_sim)
        class_output= torch.cat((x_all[0],x_all[1]),dim=1)
        for j in range(2, len((x_all))):
            class_output = torch.cat((class_output,x_all[j]),dim=1)
        x = self.sigmoid(class_output)

        return x


class Classification_GCN(nn.Module):
    def __init__(self,batch_size=1, num_classes=1,feature=64,heads=1):
        super(Classification_GCN,self).__init__()
        print('using 2020MICCAI liu for Unet-training')
        self.edge_index = torch.tensor(
        [   [1,2,0,2,1,3,2,4,3,5,3,4],
            [0,0,1,1,2,2,3,3,4,4,5,5]

            ],dtype=torch.long
                                ).cuda()


        # self.edge_index = torch.tensor(
        # [   [2,4,6,3,5,7,4,0,5,1,6,2,7,3,8,4,9,5,10,6,11,7,12,8,13,9,14,10,15,11,16,12,17,13,18,14,19,15,20,16,21,17,22,18,23,19,20,18,16,21,19,17],
        #     [0,0,0,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,22,23,23,23]
        #
        #     ],dtype=torch.long
        #                         ).cuda()
        print(self.edge_index)

        self.num_classes = num_classes
        self.batch_size = batch_size
        #self.avgpool = nn.AdaptiveAvgPool2d((16,1))
        self.conv1 = GCNConv(feature, feature)
        self.conv2 = GCNConv(feature, feature//2)
        self.conv3 = GCNConv(feature//2, feature//4,improved=True)
        self.fc = nn.ModuleList(nn.Linear(16,1)for i in range(6))
        #self.fc = nn.ModuleList(nn.Linear(16, 1) for i in range(24))
        # self.fc = nn.Linear(24*16,24)
        self.sigmoid= nn.Sigmoid()
        #self.fc2 = nn.Linear(feature//4,1)
        self.softmax = nn.Softmax(dim=2)
        #self.argmax = nn.A
    def forward(self,x):
        # x:shape[24,64]
        batch_size = x.size(0)
        # GCN classification
        # x = self.conv1(x, self.edge_index.to(x.device))
        # x = F.relu(x)
        # x = F.dropout(x,training=self.training)
        # x = self.conv2(x, self.edge_index.to(x.device))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(x,inplace=True)
        # x = self.conv3(x, self.edge_index.to(x.device))
        #
        # x = F.relu(x, inplace=True)
        # ##way1:last layer bs*node*2
        # x = self.softmax(x)
        # x = torch.argmax(x,dim=2)

        # way2:last layer bs*node*16 -> bs *channel *16 * node -> classification output
        # x = x.transpose(1, 2)
        # x = self.avgpool(x)
        # #x = torch.flatten(x, 1)
        # #x = self.fc(x)


        #
        ## GCN classification way 3
        x_gcn = x
        #print(x.device)
        x = self.conv1(x, self.edge_index)
        x = F.relu(x)
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, self.edge_index)
        #x = F.dropout(x, training=self.training)
        x = F.relu(x,inplace=True)
        x = self.conv3(x, self.edge_index)
        x = F.relu(x,inplace=True)

        #x = x_gcn+x
        x_all = []
        for i in range(x.size(1)):
            x_sim = x[:, i]
            x_sim=x_sim.squeeze().view(-1,16)

            # x_sim = self.fc(x_sim)
            x_sim = self.fc[i](x_sim)
            x_all.append(x_sim)

        class_output= torch.cat((x_all[0],x_all[1]),dim=1)
        for j in range(2, len((x_all))):
            class_output = torch.cat((class_output,x_all[j]),dim=1)
        class_output = self.sigmoid(class_output)


        # way 4:
        # x = x.view(-1, 24*16)
        #
        #         class_output = self.fc(x)
        #
        #         class_output = self.sigmoid(class_output)







        return class_output


class UNet_GCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, bilinear=True, classification=True):
        super(UNet_GCN, self).__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.classificationbranch = classification
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

                self.classifier = Classification_GCN()
                #self.classifier = Classification_sim()
        else:
            pass

    def forward(self, x):
        x1 = getattr(self, 'in{}'.format(0))(x).float()
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
        # heatmap = F.sigmoid(x[:,:self.out_channels,:,:])
        heatmap = torch.sigmoid(logits[:, :24, :, :])
        regression_x = logits[:,24:2*24,:,:]
        regression_y = logits[:,2*24:,:,:]

        if self.classificationbranch == True:
            heatmap_weight = heatmap.reshape(-1, 6, 4, heatmap.size(2), heatmap.size(3))
            heatmap_weight_con = heatmap_weight.max(2)[0]

            ## heatmap_weight_con2 = heatmap_weight.max(2)[1]
            ##heatmap_weight = torch.concat(heatmap_weight_con,heatmap_weight_con2)
            #heatmap_weight_con = heatmap
            heatmap_weight_con = heatmap_weight_con.view(heatmap_weight_con.size(0), heatmap_weight_con.size(1),
                                                         heatmap_weight_con.size(2) * heatmap_weight_con.size(3))
            # using lowest features
            x_cl = x.view(x.size(0), x.size(1), x.size(2) * x.size(3)).transpose(1, 2)
            class_weight = torch.bmm(heatmap_weight_con, x_cl) / x.size(1)
            class_output = self.classifier(class_weight).float()

        return heatmap, regression_x, regression_y, class_output


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pytorch_wavelets import DWTForward, DWTInverse
# from torch.autograd import Variable
# from torch_geometric.nn import GCNConv,GATConv
# class dwise(nn.Module):
#     def __init__(self, inChans, kernel_size=3, stride=1, padding=1):
#         super(dwise, self).__init__()
#         self.conv1 = nn.Conv2d(inChans, inChans, kernel_size=kernel_size,
#                                stride=stride, padding=padding, groups=inChans)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         return out
#
#
# class pwise(nn.Module):
#     def __init__(self, inChans, outChans, kernel_size=1, stride=1, padding=0):
#         super(pwise, self).__init__()
#         self.conv1 = nn.Conv2d(
#             inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         return out
#
#
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.dwise1 = dwise(in_channels)
#         self.dwise2 = dwise(mid_channels)
#         self.pwise1 = pwise(in_channels, mid_channels)
#         self.pwise2 = pwise(mid_channels, out_channels)
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu1 = nn.LeakyReLU(inplace=True)
#         self.relu2 = nn.LeakyReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.pwise1(self.dwise1(x))
#         x = self.relu1(self.bn1(x))
#         x = self.pwise2(self.dwise2(x))
#         x = self.relu2(self.bn2(x))
#         return x
#
#
# class Down(nn.Module):
#     """Downscaling with DWT then double conv"""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         #xfm =
#         #self.maxpool = nn.MaxPool2d(2)
#         self.pooling =DWTForward(J=1, mode='zero', wave='haar').cuda()
#         self.conv_DMT1 = nn.Conv2d(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn_DMT1=nn.BatchNorm2d(out_channels, affine=True)
#         self.relu_DMT1=nn.ReLU()
#         self.conv = DoubleConv(out_channels, out_channels )
#     def _transformer(self,DWT1_yl,DMT1yh):
#         list_tensor = []
#         for i in range(3):
#             list_tensor.append(DMT1yh[0][:,:,i,:,:])
#         list_tensor.append(DWT1_yl)
#         return torch.cat(list_tensor, 1 )
#
#     def forward(self, x):
#         #input tensor bs 64* h*w
#         #DMT_p = x
#         DMT1_yl, DMT1_yh = self.pooling(x)
#         DMT1 = self._transformer(DMT1_yl,DMT1_yh)
#
#         out = self.relu_DMT1(self.bn_DMT1(self.conv_DMT1(DMT1)))
#         out = self.conv(out)
#         return out
#
#         #return self.conv(self.maxpool(x))
#
#
#
#
#
# class Up(nn.Module):
#     """Upscaling then double conv"""
#
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#
#         # if bilinear, use the normal convolutions to reduce the number of channels
#
#         self.IDMT = DWTInverse(wave='haar').cuda()
#         self.conv_IDMT = nn.Conv2d(in_channels=in_channels//2,out_channels=in_channels*2,kernel_size=3,stride=1,padding=1)
#         self.bn_IDMT = nn.BatchNorm2d(in_channels*2,affine=True)
#         self.relu_IDMT = nn.ReLU()
#
#
#         if bilinear:
#             self.up = nn.Upsample(
#                 scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(
#                 in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(
#                 in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(
#                 in_channels, out_channels)
#
#     def _Itransformer(self,out):
#         yh = []
#         C  = int(out.shape[1]/4)
#
#         y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
#         yl = y[:, :, 0].contiguous()
#         yh.append(y[:, :, 1:].contiguous())
#
#         return yl,yh
#
#     def forward(self, x1, x2):
#
#         x1 = self.relu_IDMT(self.bn_IDMT(self.conv_IDMT(x1)))
#         x1 = self._Itransformer(x1)
#         x1 = self.IDMT(x1)
#
#
#         #x1 = self.up(x1)
#         # input is CHW
#         #diffY = x2.size()[2] - x1.size()[2]
#         #diffX = x2.size()[3] - x1.size()[3]
#
#         #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         #diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         #concate
#         x = torch.cat([x1,x2],1)
#         #add
#         #x = x1+x2
#         return self.conv(x)
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#                                 nn.ReLU(),
#                                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.conv1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
#                                    nn.BatchNorm2d(1, momentum=0.01, affine=True))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
# class Classification_only(nn.Module):
#     def __init__(self,batch_size=1, num_classes=6, baseline=True, using_attention= True):
#         super(Classification_only,self).__init__()
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.SpatialAttention =SpatialAttention()
#         self.maxpool = nn.AdaptiveAvgPool2d((1,1))
#         #print('using MLT training')
#         print('-----using MLT baseline-----')
#         self.fc = nn.Linear(64,self.num_classes)
#         self.relu = nn.ReLU()
#         self.mode = baseline
#         self.using_att = using_attention
#         self.sig = nn.Sigmoid()
#
#         #self.fc2 = nn.Linear(batch_size,)
#     def forward(self,x):
#         batch_size = x.size(0)
#         if self.using_att:
#             x_weight = self.SpatialAttention(x)
#             x = x*x_weight
#
#         x = self.maxpool(x)
#         x=x.squeeze().view(batch_size,-1)
#         x = self.fc(x)
#         x = self.sig(x)
#         return x
#
#
#
# class Classification_sim(nn.Module):
#     def __init__(self,batch_size=1, num_classes=1):
#         super(Classification_sim,self).__init__()
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         print('using AGL for training')
#         self.avgpool2d = nn.AdaptiveMaxPool2d((1,1))
#         self.avgpool1d = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.ModuleList(nn.Linear(64,1)for i in range(6))
#         #self.fc1 =nn.Linear(64,64)
#         #self.dynamic = nn.Linear(64,128)
#         self.relu = nn.ReLU(inplace=True)
#         #self.fc = nn.Linear(64,1)
#         self.sigmoid = nn.Sigmoid()
#         #self.fc2 = nn.Linear(batch_size,)
#     def forward(self,x):
#         x_all = []
#         if len(x.size())== 4:
#             x = self.avgpool2d(x)
#         for i in range(x.size(1)):
#             #x_sim = x[:, i]
#         #     # #anmatual attention resnet
#         #     if i< x.size(1)-2:
#         #         x_sim= x[:,i]
#         #         # parameters = self.dynamic(x)
#         #         # param1 = parameters[:, :, :64].contiguous().view(-1, 64, 24)
#         #         # param2 = parameters[:, :, 64:].contiguous().view(-1, 24, 64)
#         #         # features = torch.bmm(x, param1)
#         #         # features = torch.bmm(features, param2)
#         #         # x_nearest = features[:,i+2]
#         #         #print(x_sim.size())
#         #         #print(x_nearest.size())
#         #         x_nearest = x[:, i + 2]
#         #         x_diff = x_sim-x_nearest
#         #         x_sim = x_sim+x_diff
#         #     else:
#         #         x_sim =x[:,i]
#         #         # parameters = self.dynamic(x)
#         #         # param1 = parameters[:, :, :64].contiguous().view(-1, 64, 24)
#         #         # param2 = parameters[:, :, 64:].contiguous().view(-1, 24, 64)
#         #         # features = torch.bmm(x, param1)
#         #         # features = torch.bmm(features, param2)
#         #         # x_nearest = features[:,i-2]
#         #         x_nearest = x[:, i - 2]
#         #         x_diff = x_sim - x_nearest
#         #         x_sim = x_sim+x_diff
#             #x_sim = self.relu(self.fc1(x_sim))
#
#             x_sim = x[:, i]
#             x_sim=x_sim.squeeze().view(-1,64)
#             x_sim = self.fc[i](x_sim)
#             x_all.append(x_sim)
#         class_output= torch.concat((x_all[0],x_all[1]),dim=1)
#         for j in range(2, len((x_all))):
#             class_output = torch.concat((class_output,x_all[j]),dim=1)
#         x = self.sigmoid(class_output)
#
#         return x
#
# class Classification_GCN(nn.Module):
#     def __init__(self,batch_size=1, num_classes=1,feature=64,heads=1):
#         super(Classification_GCN,self).__init__()
#         print('using U2GCN for training')
#         self.edge_index = torch.tensor(
#         [   [1,2,0,2,1,3,2,4,3,5,3,4],
#             [0,0,1,1,2,2,3,3,4,4,5,5]
#
#             ],dtype=torch.long
#                                 ).cuda()
#
#
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#         self.conv1 = GCNConv(feature, feature)
#         self.conv2 = GCNConv(feature, feature)
#         self.conv3 = GCNConv(feature, feature,improved=True)
#         #self.conv3 = GATConv(feature, feature,heads= heads)
#         #self.conv4 = GATConv(feature*heads, feature)
#         self.fc = nn.ModuleList(nn.Linear(64,1)for i in range(6))
#         # self.fc = nn.Linear(64,1)
#         #self.fc2 = nn.Linear(128,6)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self,x):
#         # x:shape[24,64]
#
#         ## GCN classification
#         # x = self.conv1(x, self.edge_index.to(x.device))
#         # x = F.relu(x)
#         # #x = F.dropout(x,training=self.training)
#         # x = self.conv2(x, self.edge_index.to(x.device))
#         # #x = F.dropout(x, training=self.training)
#         # x = F.relu(x,inplace=True)
#         # x = self.conv3(x, self.edge_index.to(x.device))
#         # x = F.relu(x,inplace=True)
#         # x = x.view(-1, 64*6)
#         # x = self.fc(x)
#         # x = F.relu(x, inplace=True)
#         # x = self.fc2(x)
#         #
#
#
#         # #self.fc = nn.Linear(64, 6)
#         # x = self.sigmoid(x)
#         # attention_distribution = []
#         # #calulate edge_weight using cosine similarity
#         # for edge in range(x.size(1)):
#         #     if edge == 0 :
#         #         attention_score = torch.cosine_similarity(x[:,edge],x[:,edge+1],dim=1)
#         #         attention_distribution.append(attention_score)
#         #         attention_score = torch.cosine_similarity(x[:,edge],x[:,edge+2],dim=1)# 计算每一个元素与给定元素的余弦相似度
#         #         attention_distribution.append(attention_score)
#         #     elif edge ==5 :
#         #         attention_score = torch.cosine_similarity(x[:,edge],x[:,edge-1],dim=1)
#         #         attention_distribution.append(attention_score)
#         #         attention_score = torch.cosine_similarity(x[:,edge],x[:,edge-2],dim=1)# 计算每一个元素与给定元素的余弦相似度
#         #         attention_distribution.append(attention_score)
#         #     else:
#         #         attention_score = torch.cosine_similarity(x[:, edge], x[:, edge+1], dim=1)
#         #         attention_distribution.append(attention_score)
#         #         attention_score = torch.cosine_similarity(x[:, edge], x[:, edge-1], dim=1)  # 计算每一个元素与给定元素的余弦相似度
#         #         attention_distribution.append(attention_score)
#         #
#         # attention_distribution = torch.Tensor(attention_distribution)
#         #
#         #attention_distribution = attention_distribution / torch.sum(attention_distribution, 0)
#
#
#
#
#         #GCN weighted
#         x_gcn = self.conv1(x, self.edge_index.to(x.device))
#         x_gcn = F.relu(x_gcn)
#         # x_gcn = F.dropout(x_gcn,training=self.training)
#         x_gcn = self.conv2(x_gcn,self.edge_index.to(x.device))
#         x_gcn = F.relu(x_gcn)
#         # x_gcn = F.dropout(x_gcn, training=self.training)
#         x_gcn = self.conv3(x_gcn, self.edge_index.to(x.device))
#         x_gcn = F.relu(x_gcn)
#
#
#         # resident
#         #x = x + x_gcn
#         x_all = []
#         for i in range(x_gcn.size(1)):
#             x_sim = x_gcn[:, i]
#             x_sim=x_sim.squeeze().view(-1,64)
#
#             # x_sim = self.fc(x_sim)
#             x_sim = self.fc[i](x_sim)
#             x_all.append(x_sim)
#
#         class_output= torch.concat((x_all[0],x_all[1]),dim=1)
#         for j in range(2, len((x_all))):
#             class_output = torch.concat((class_output,x_all[j]),dim=1)
#         class_output = self.sigmoid(class_output)
#
#         #x = F.dropout(x_gcn,training=self.training)
#         #x = self.sigmoid(x).squeeze(dim=2)
#
#
#
#
#
#         return class_output

# class Classification(nn.Module):
#     def __init__(self,batch_size=1, num_classes=1):
#         super(Classification,self).__init__()
#         self.num_classes = num_classes
#         self.batch_size = batch_size
#
#         #self.maxpool = nn.AdaptiveMaxPool2d((1,1))
#         self.fc = nn.Linear(64,1)
#         self.softmax = nn.Sigmoid()
#         #self.fc2 = nn.Linear(batch_size,)
#     def forward(self,x):
#         x = torch.sum(x,dim=(2,3))
#         x=x.squeeze().view(-1,64)
#         x = self.fc(x)
#         #x.reshape(self.batch_size,self.num_classes)
#         x = self.softmax(x)
#         return x
# class OutConv_Sigmoid(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv_Sigmoid, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         return self.sigmoid(self.conv(x))
#
# class U2NetGCN(nn.Module):
#     def __init__(self, in_channels=1, out_channels=24, bilinear=True,classification=True,classification_gcn =True,classification_only=True):
#         super(U2NetGCN, self).__init__()
#         if not isinstance(in_channels, list):
#             in_channels = [in_channels]
#         if not isinstance(out_channels, list):
#             out_channels = [out_channels]
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.n_classes = out_channels
#         self.bilinear = bilinear
#         #self.task_num = len(in_channels)
#
#         for i, (n_chan, n_class) in enumerate(zip(in_channels, out_channels)):
#             setattr(self, 'in{i}'.format(i=i), OutConv(n_chan, 64))
#             setattr(self, 'out{i}'.format(i=i), OutConv(64, n_class*3))
#         #self.tans = OutConv(1,64)
#         #self.outp = OutConv(64,24)
#         self.conv = DoubleConv(64, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor, )
#         # self.up1 = Up(512, 512 // factor, bilinear, )
#         # self.up2 = Up(256, 256 // factor, bilinear, )
#         # self.up3 = Up(128, 128 // factor, bilinear, )
#         # self.up4 = Up(64, 64, bilinear, )
#
#         self.up1 = Up(1024, 512 // factor, bilinear, )
#         self.up2 = Up(512, 256 // factor, bilinear, )
#         self.up3 = Up(256, 128 // factor, bilinear, )
#         self.up4 = Up(128, 64, bilinear, )
#
#         #self.outc = OutConv_Sigmoid(64, out_channels)
#         #self.outc_regression_x = OutConv(64, out_channels)
#         #self.outc_regression_y = OutConv(64, out_channels)
#         self.softmax = nn.Softmax(dim=1)
#         self.sig = nn.Sigmoid()
#         self.classification_only = classification_only
#         self.classificationbranch = classification
#
#
#         if classification:
#             if classification_only:
#                 self.classifier_only = Classification_only()
#
#
#             elif classification_gcn:
#                 self.classifier = Classification_GCN()
#             else:
#                 self.classifier = Classification_sim()
#
#         else:
#             pass
#
#         # self.edge_index = torch.tensor(
#         # [
#         #     [0,0,0,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,22,23,23,23],
#         #     [2,4,6,3,5,7,4,0,5,1,6,2,7,3,8,4,9,5,10,6,11,7,12,8,13,9,14,10,15,11,16,12,17,13,18,14,19,15,20,16,21,17,22,18,23,19,20,18,16,21,19,17]
#         #     ],dtype=torch.long
#         #                         ).cuda()
#
#         # self.edge_index = torch.tensor(
#         # [   [2,4,6,3,5,7,4,0,5,1,6,2,7,3,8,4,9,5,10,6,11,7,12,8,13,9,14,10,15,11,16,12,17,13,18,14,19,15,20,16,21,17,22,18,23,19,20,18,16,21,19,17],
#         #     [0,0,0,1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20,21,21,22,22,22,23,23,23]
#         #
#         #     ],dtype=torch.long
#         #                         ).cuda()
#
#         self.edge_index = torch.tensor(
#         [   [1,2,0,2,1,3,2,4,3,5,3,4],
#             [0,0,1,1,2,2,3,3,4,4,5,5]
#
#             ],dtype=torch.long
#                                 ).cuda()
#         self.pooling = nn.AdaptiveMaxPool2d((32,32))
#     def forward(self, x):
#         x1 = getattr(self, 'in{}'.format(0))(x).float()
#         #x1 = self.tans(x1)
#         x1 = self.conv(x1)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         #x = self.outp(x)
#         #device = torch.device(x)
#         #keypoints branch
#         logits = getattr(self, 'out{}'.format(0))(x)
#         # heatmap = F.sigmoid(x[:,:self.out_channels,:,:])
#         heatmap = torch.sigmoid(logits[:, :24, :, :])
#         regression_x = logits[:,24:2*24,:,:]
#         regression_y = logits[:,2*24:,:,:]
#
#
#         #heatmap = self.sig(logits)
#         #
#         #classification branch
#
#         if self.classificationbranch == True:
#             if self.classification_only == True:
#                 class_output = self.classifier_only(x)
#             else:
#
#                 #heatmap_weight = self.pooling(heatmap)
#                 #heatmap_weight = heatmap
#                 heatmap_weight = heatmap.reshape(-1,6,4,heatmap.size(2),heatmap.size(3))
#                 heatmap_weight_con = heatmap_weight.max(2)[0]
#                 #heatmap_weight_con2 = heatmap_weight.max(2)[1]
#                 #heatmap_weight = torch.concat(heatmap_weight_con,heatmap_weight_con2)
#
#                 heatmap_weight_con = heatmap_weight_con.view(heatmap_weight_con.size(0),heatmap_weight_con.size(1),
#                                               heatmap_weight_con.size(2)*heatmap_weight_con.size(3))
#                 #using lowest features
#                 x_cl = x.view(x.size(0),x.size(1), x.size(2)*x.size(3)).transpose(1,2)
#                 class_weight = torch.bmm(heatmap_weight_con,x_cl)/x.size(1)
#                 class_output = self.classifier(class_weight)
#
#
#             return heatmap, regression_x, regression_y, class_output
#         else:
#             return heatmap, regression_x, regression_y
#
#         # v1.0
#         #heatmap_nor_expand = torch.unsqueeze(heatmap_nor,dim=2)
#         #x_expand = torch.unsqueeze(x,dim=1)
#         #heatmap_nor_expand = heatmap_nor_expand.expand(2,24,64,512,512)
#         #x_expand = x_expand.expand(2,24,64,512,512)
#         # #tensor_weight = torch.mul(heatmap_nor_expand,x_expand)
#         #class_output = self.classifictaionbranch(tensor_weight)
#         #return heatmap,regression_x,regression_y



from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.optim as optim
#from models import KFSGNet
class tempDataset(Dataset):
    def __init__(self):
        self.X = np.random.randn(8,1,512,512)
        self.Y = np.random.randn(8,24,512,512)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        # 这里返回的时候不要设置batch_size
        return self.X[item],self.Y[item]

if __name__ == '__main__':
    from torch.nn import MSELoss
    critical = MSELoss()

    dataset = tempDataset()
    dataLoader = DataLoader(dataset=dataset,batch_size=2)
    #shg = (in_channels=1, out_channels=24).cuda()
    shg = MMWUNet(in_channels=1,out_channels=24,classification=True,classification_gcn=True,classification_only=False)

    optimizer = optim.SGD(shg.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-4)

    for e in range(200):
        for i,(x,y) in enumerate(dataLoader):
            x = Variable(x,requires_grad=True).float().cuda()
            y = Variable(y).float().cuda()
            y_pred,class_pred = shg.forward(x)
            loss = critical(y_pred[0],y[0])
            print('loss : {}'.format(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()