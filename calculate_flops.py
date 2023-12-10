import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint
import transform_new
from sklearn import metrics
from dataloader_new import KFDataset
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from vit_model import vit_base_patch16_224_in21k as creare_model
import random

# Evaluation
from train_eval_class import evaluate_one
from train_eval_class_backup import evaluate_all
from fvcore.nn import FlopCountAnalysis
from torchstat import  stat
# Five fold validations
from sklearn.model_selection import KFold

##Model Selection
from MMWUet import MMWUNet
from U2Netnew import U2Net
from Unet_Maw import UNet
from ED_optional import UResnet,BottleNeck,BasicBlock
from Unet_base import UNet_base
from torchvision.models.resnet import resnet50

from torchprofile import profile_macs
from Unet_concate import UNet_conc
from Unet_GCN import UNet_GCN
from Y_Net import YNet
from Unet_dual import UNet_Dual
# import thop
# from thop import profil
##Loss Definition
from loss import KpLoss ,CLALoss
import glob


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--keypoint_model", type=str, default='our+module2',
                        help="the model name"
                             "2020shen"
                             "YNet"
                             "2020Wang "
                             "2019MICCAI")



    parser.add_argument("--sigma", type=float, default=10.0,
                        help="the sigma of generated heatmaps.")

    parser.add_argument('--keypoint_batch_size', type=int, default=4,
                        help="The batch size, default: 4")

    parser.add_argument('--keypoint_loss', type=str, default='MSELoss',
                        help="The keypoint loss function name, MSELoss, MSEDSLoss, CrossEntropyLoss.")

    parser.add_argument("--gpu_index", type=int, default=[0,1],
                        help="gpu index")
    parser.add_argument("--attention", default=False,help="saptial attention")
    parser.add_argument("--keypoint_epochs", type=int, default=100,
                        help="max number of epochs, default: 100")
    parser.add_argument("--seed", type=int, default=42,
                        help="randomseed, default: 0")
    parser.add_argument("--keypoint_model_dir", type=str, default='./Checkpoints_final/',
                        help="saving keypoint model_dir")

    parser.add_argument('--keypoint_learning_rate', type=float, default=0.01,
                        help="The initial learning rate, default: 5e-3"
                             "for diffierent model:Unet_gcn:0.0005"
                             "Ynet :0.0005 Ynet+sap:0.001"
                             "HeadlocNet 0.0005 Concate+sap0.0001"
                             "cUnet:0.005"
                             "ResNet:0.001,+sap0.0005"
                             "ablation for all:0.005")




    args = parser.parse_args()

    config = dict()
    config['lr'] = args.keypoint_learning_rate
    config['momentum'] = 0.009
    config['weight_decay'] = 1e-4
    config['epoch_num'] = args.keypoint_epochs
    config['batch_size'] = args.keypoint_batch_size

    config['sigma'] = args.sigma
    config['debug_vis'] = False
    config['device'] = args.gpu_index
    config['train_fname'] = ''
    config['test_fname'] = ''
    config['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_test/'
    config['train_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'

    config['path_label'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/'
    config['path_label_train'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/train_json/'
    config['is_test'] = False
    config['lr_steps'] = [60, 80]
    # config['lr_steps'] = [80, 90] # fir sa
    config['lr_gamma'] = 0.5

    config['amp'] = True
    config['save_freq'] = 5
    config['checkout'] = args.keypoint_model_dir + args.keypoint_model +'/'
    config['start_epoch'] = 0
    config['load_pretrained_weights'] = False
    config['eval_freq'] = 50
    config['debug'] = False
    config['featurename2id'] = {
        'C2_TR': 0,
        'C2_TL': 1,
        'C2_DR': 2,
        'C2_DL': 3,
        'C3_TR': 4,
        'C3_TL': 5,
        'C3_DR': 6,
        'C3_DL': 7,
        'C4_TR': 8,
        'C4_TL': 9,
        'C4_DR': 10,
        'C4_DL': 11,
        'C5_TR': 12,
        'C5_TL': 13,
        'C5_DR': 14,
        'C5_DL': 15,
        'C6_TR': 16,
        'C6_TL': 17,
        'C6_DR': 18,
        'C6_DL': 19,
        'C7_TR': 20,
        'C7_TL': 21,
        'C7_DR': 22,
        'C7_DL': 23,
    }



    images1 = sorted(glob.glob(os.path.join(config['train_image_path'], '*.jpg')))
    labels1 = sorted(glob.glob(os.path.join(config['path_label_train'], '*_jpg_Label.json')))
    floder = KFold(n_splits=5, random_state=42, shuffle=True)
    data_dicts1 = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(images1, labels1)]
#
    train_files = []
    test_files = []
    for k, (Trindex, Tsindex) in enumerate(floder.split(data_dicts1)):
        train_files.append(np.array(data_dicts1)[Trindex].tolist())
        test_files.append(np.array(data_dicts1)[Tsindex].tolist())


    ### save fold
    # df = pd.DataFrame(data=train_files, index=['0', '1', '2', '3', '4'])
    # df.to_csv('./txt/Kfold/train_patch.csv')
    # df1 = pd.DataFrame(data=test_files, index=['0', '1', '2', '3', '4'])
    # df1.to_csv('./txt/Kfold/test_patch.csv')





    pprint.pprint(config)
    seed_torch(args.seed)
    print('using fixed seed ,',args.seed)
    cudnn.benchmark = True


    #model selection
    if args.keypoint_model == '2020shen':
        net = UNet_GCN(in_channels=1, out_channels=24, bilinear=True, classification=True)
        net_stat = UNet_GCN(in_channels=1, out_channels=24, bilinear=True, classification=True)
        config['lr'] = 0.0005
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/2020shen/'
    elif args.keypoint_model == 'YNet':
        # best parameter lr=0.0005 milestone=[60,80]
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/YNet/'
        if args.attention:
            net = YNet(in_channels=1, out_channels=24, bilinear=True, classification=True, attention=True)
            config['lr'] = 0.0001

        else:
            net = YNet(in_channels=1, out_channels=24, bilinear=True, classification=True, attention=False)
            config['lr'] = 0.0005
            #config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/YNet/'
    elif args.keypoint_model == '2020Wang':
        # best parameter lr = 0.0005 milestone=[60,80]
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/2020Wang/'
        if args.attention:
            net = UNet_conc(in_channels=1, out_channels=24, bilinear=True, classification=True,attention=True)
            config['lr'] = 0.0001
        else:
            net = UNet_conc(in_channels=1, out_channels=24, bilinear=True, classification=True,attention=False)
            config['lr'] = 0.0005

    elif args.keypoint_model == 'Mlt_highest_sap':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/Mlt_highest_sap/'
        # best parameter lr =.0.001
        if args.attention:
            net = UResnet(block=BottleNeck, layers=[3, 4, 6, 3], num_classes=24,attention = True)
            config['lr'] = 0.0005

        else:
            net = UResnet(block=BottleNeck, layers=[3, 4, 6, 3], num_classes=24, attention=False)
            config['lr'] = 0.001

    elif args.keypoint_model =='baseline':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/baseline/'
        if args.attention:
            net = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=True,attention=True)
            config['lr'] = 0.005
            net_stat = net

        else:
            net = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=True,attention=False)
            config['lr'] = 0.005
            net_stat = net
    elif args.keypoint_model == 'our+module1':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+ACL/'
        net = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=False)
        config['lr'] = 0.005
        net_stat = net
    elif args.keypoint_model == 'our+module2':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+WTS/'

        ## 0804add
        net = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=True)
        #config['lr'] = 0.005
        net_stat = net
    elif args.keypoint_model == 'our+module1+2':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/MAW/'
        net = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=False)
        net_stat = net
        config['lr'] = 0.005

    elif args.keypoint_model == 'our+module1+3':
        net = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=True,
                    classification_only=False)
        config['lr'] = 0.005
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+ACL+GCN/'
    elif args.keypoint_model== 'our_all':
        net = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=True,
                    classification_only=False)
        config['lr'] = 0.005
        #config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/ALL/'

    else:
        net = UNet_base(in_channels=1, out_channels=24, bilinear=True, classification=False)
        net_stat = net
        config['lr'] = 0.005
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/2019MICCAI/'
    print('the number of trainable parameters: %d' % get_number_of_learnable_parameters(net))
    print('Initial learning rate :',config['lr'])
    config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/SA+WTS+BAM/{}'.format(
        args.seed)
    print ('Saving checkpoint to',config['checkout'])

    #net = UNet_base(in_channels=1, out_channels=24, bilinear=True, classification=False)
    #net = YNet(in_channels=1, out_channels=24, bilinear=True, classification=True, attention=True)
    # from torchstat import stat
    # import torchvision.models as models
    # net =
    print('the number of trainable parameters: %d' % get_number_of_learnable_parameters(net))
    gpus = [g for g in range(torch.cuda.device_count())]
    print(len(gpus))
    if len(gpus) > 1:
        net = nn.DataParallel(net, device_ids=gpus)

    #net = net
    from torchstat import stat
    import torchvision.models as models

   # net = torchvision.models.densenet121(pretrained=True, num_classes=6)
   #  model = models.densenet121(pretrained=True, num_classes=6)
   #  stat(model, (1, 512, 512))


    input_data = torch.randn(1,1,512,512)
    macs = profile_macs(net,input_data)
    print(macs/(10**9))
    F1 = FlopCountAnalysis(net,input_data)
    print(F1.total())


    stat(net_stat,(1,512,512))
