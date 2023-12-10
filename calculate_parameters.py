##Model Selection
from MMWUet import MMWUNet
from U2Netnew import U2Net
from Unet_Maw import UNet
from ED_optional import UResnet,BottleNeck,BasicBlock
from Unet_base import UNet_base
import numpy as np


from Unet_concate import UNet_conc
from Unet_GCN import UNet_GCN
from Y_Net import YNet



def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

##testing
    # Generalize between backbone
# net  = UNet(in_channels=1, out_channels=24,bilinear=True,classification=True)
# print('inital Unet_Maw')
# print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))
#
#
# print('inital Unet')
# net = UNet_base(in_channels=1, out_channels=24, bilinear=True, classification=True)
# print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))
#
# print('inital U2net')
# net = U2Net(in_channels=1, out_channels=24,classification=True,classification_gcn=False,classification_only=True)
# print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))
# ## Our proposed method based on U2net
#
#
# print('inital U2net+APL')
# net = U2Net(in_channels=1, out_channels=24,classification=True,classification_gcn=False,classification_only=False)
# print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))
#
#
# print('inital Maw')
# net = MMWUNet(in_channels=1,out_channels=24,classification=True,classification_gcn=True,classification_only=False)
# print('the number of trainable parameters: %2f M' %(get_number_of_learnable_parameters(net)/(10**6)))
# ##2020 MICCAI Liu
# net = UNet_GCN(in_channels=1, out_channels=24, bilinear=True, classification=True)
# print('inital Unet_GCN')
# print('the number of trainable parameters: %2f M' %(get_number_of_learnable_parameters(net)/(10**6)))
#     ##2022 MIA Wang
# net = UNet_conc(in_channels=1, out_channels=24, bilinear=True, classification=True)
# print('inital 2020MIA' )
# print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))
#     ## 2019 Y-Net
# net = YNet(in_channels=1, out_channels=24, bilinear=True, classification=True)
# print('inital Ynet' )
# print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))
#     # 2020 MIA Wang2.0
# net = UResnet(block=BottleNeck, layers=[3, 4, 6, 3], num_classes=24)
# print('UResnet' )
# print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))


## our_all

net =MMWUNet(in_channels=1, out_channels=24,classification=True,classification_gcn=True,classification_only=False)
print('inital baseline')
print('the number of trainable parameters: %2f M' % (get_number_of_learnable_parameters(net)/(10**6)))