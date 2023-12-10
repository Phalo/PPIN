#coding=utf-8
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint
import transform_spine
from sklearn import metrics
from dataloader_class import KFDataset
#from models import KFSGNet
import matplotlib.pyplot as plt
import os
import argparse
from eval_ResNet import evaluate_one ,evaluate_Trans
#from network import UNet_Pretrained
from MMWUet import MMWUNet
import torchvision
from U2Netnew import U2Net
from loss import KpLoss ,CLALoss
from sklearn.model_selection import KFold
import glob
import time
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
#from loss import CLALoss
from SEnet import SEResNet101
from resnet_base import resnet18
from Spine_transformer import  SpineTransformer,build
import random
from vit_model import vit_base_patch16_224_in21k as creare_model
parser = ArgumentParser(description=__doc__,
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--keypoint_model", type=str, default='Spine_T',
                    help="the model name"
                         "ResNet50"
                         "DenseNet121"
                         "SENet101 "
                         "2019MICCAI")

# parser.add_argument("--data_dir", type=str, default='../data',
#                     help="the data dir")

parser.add_argument("--sigma", type=float, default=10.0,
                    help="the sigma of generated heatmaps.")
parser.add_argument("--seed", type=int, default=0,
                    help="the sigma of generated heatmaps.")

parser.add_argument('--keypoint_batch_size', type=int, default=4,
                    help="The batch size, default: 4")
parser.add_argument("--keypoint_model_dir", type=str, default='./Checkpoints_final/',
                    help="saving keypoint model_dir")

parser.add_argument('--keypoint_learning_rate', type=float, default=1e-4
                    ,
                    help="The initial learning rate, default: 5e-3")

parser.add_argument('--weights', type=str,
                    default='/public/huangjunzhang/Facetjoints/ViTweight/vit_base_patch16_224_in21k.pth',
                    help="load pretrained weight")

## transformer
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--enc_layers', default=1, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=1, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=512, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.5, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=24, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--freeze-layers', type=bool, default=True)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")
parser.add_argument('--backbone', default='resnet18', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")

parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

args = parser.parse_args()

config = dict()
config['lr'] = args.keypoint_learning_rate

# config['lr'] = 0.005

config['momentum'] = 0.009
config['weight_decay'] = 1e-4
config['epoch_num'] = args.epochs
config['batch_size'] = args.keypoint_batch_size

config['sigma'] = 10.0
config['debug_vis'] = False
config['device'] = "cuda:1"
config['train_fname'] = ''
config['test_fname'] =''
#config ['path_image'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'
config ['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_test/'
config ['train_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'

config['path_label'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/'
config['path_label_train'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/train_json/'
#config['json_path']='/public/huangjunzhang/test/keypoints_train.json'
config['is_test'] = False
config['lr_steps'] = [100*0.6, 100*0.8]
config['lr_gamma'] = 0.5
config['amp'] = True
config['save_freq'] = 5
#config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ResNet50/'
config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/'
#config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/'
config['start_epoch'] = 0
config['load_pretrained_weights'] = False
config['eval_freq'] = 50
config['debug'] = False
config['featurename2id'] = {
    'C2_TR':0,
    'C2_TL':1,
    'C2_DR':2,
    'C2_DL':3,
    'C3_TR':4,
    'C3_TL':5,
    'C3_DR':6,
    'C3_DL':7,
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





def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True









if __name__ == '__main__':
    #init_distributed_mode(args=args)
    images1 = sorted(glob.glob(os.path.join(config['train_image_path'], '*.jpg')))
    labels1 = sorted(glob.glob(os.path.join(config['path_label_train'], '*_jpg_Label.json')))
    folder = KFold(n_splits=5, random_state=42, shuffle=True)
    data_dicts1 = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(images1, labels1)]
    #
    train_files = []
    test_files = []
    for k, (Trindex, Tsindex) in enumerate(folder.split(data_dicts1)):
        train_files.append(np.array(data_dicts1)[Trindex].tolist())
        test_files.append(np.array(data_dicts1)[Tsindex].tolist())
    #device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    pprint.pprint(config)
    torch.manual_seed(args.seed)


    cudnn.benchmark = True

    if args.keypoint_model == 'DenseNet121':
        print('using pretrained densenet')
        net = torchvision.models.densenet121(pretrained=True,num_classes=6)
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/DenseNet121/'
        #config['lr'] = 0.001
    elif args.keypoint_model == 'SENet101':
        net = SEResNet101()
        config['checkout'] ='/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/SEnet101/'
        #config['lr'] = 0.001

    elif args.keypoint_model == 'Spine_T':
        print('using spine transformer for training')
        net, criterion, postprocessors = build(args)
        config['checkout'] ='/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/Spine/'
        #config['lr'] = 0.001
    elif args.keypoint_model == 'Vit':
        print('using vision transformer')
        net = creare_model(num_classes=6, has_logits=False)

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if net.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(net.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in net.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
    else:
        net = torchvision.models.resnet50(pretrained=True,num_classes=6)
        config['checkout'] ='/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ResNet50/'
        # config['lr'] = 0.001

        #net = torchvision.models.se
    print('Initial learning rate :',config['lr'])
    print ('Saving checkpoint to',config['checkout'])

    gpus = [g for g in range(torch.cuda.device_count())]
    if len(gpus) > 1:
        net = nn.DataParallel(net, device_ids=gpus)
    net = net.cuda()


    #criterion = torch.nn.BCELoss()
    #criterion = KpLoss()
    #class_criterion = CLALoss()
    class_criterion = torch.nn.BCELoss(reduction='mean')

    coord_criterion = nn.SmoothL1Loss(reduction='mean')
    coord_criterion_none = nn.SmoothL1Loss(reduction='none')
    #coord_criterion = nn.MSELoss(reduction='mean')

    #criterion = nn.BCELoss()
    #optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'] , weight_decay=config['weight_decay'])
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_steps'], gamma=config['lr_gamma'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        total_steps=300,
        pct_start=0.7,
        epochs=config['epoch_num'],
        last_epoch=0 - 1,
    )

    # optimizer = optim.RMSprop(net.parameters(),lr=config['lr'],
    #                                 weight_decay=config['weight_decay'],
    #                                 momentum=config['momentum'])
    # 定义 Dataset

    data_transforms = {
        "train": transform_spine.Compose([
                                     #transform_new.Resize(H=224,W=224),
                                     transform_spine.RandomHorizontalFlip(0.5),
                                     transform_spine.Blur(),
                                     #transforms.Brightness(),
                                     transform_spine.ToTensor(),

                                     # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ]),

        "val" : transform_spine.Compose([transform_spine.ToTensor(),
                                    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])
    }
    trainDataset = KFDataset(config , mode='train', transforms=data_transforms["train"],fold=train_files[0])
    # 定义 data loader
    trainDataLoader = DataLoader(trainDataset,config['batch_size'],True,num_workers=8)


    sample_num = len(trainDataset)
    print(sample_num)


    valDataset = KFDataset(config , mode='test', transforms=data_transforms["val"],fold=test_files[0])
    valDataLoader = DataLoader(valDataset, 1, False, num_workers=8)

    TvalDataset = KFDataset(config , mode='train', transforms=data_transforms["val"],fold=test_files[0])
    TvalDataLoader = DataLoader(TvalDataset, 1, False, num_workers=8)


    if config['load_pretrained_weights']:
         if (config['checkout'] != ''):
             print("load dict from checkpoint")
             net.load_state_dict(torch.load(config['checkout']))
    train_loss = []
    vali_loss = []
    best_auc = 0
    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):
        net.float().cuda()
        net.train()
        #metric_logger = utils.MetricLogger(delimiter="  ")
        #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        #header = 'Epoch: [{}]'.format(epoch)

        print("current learn rate:", optimizer.state_dict()['param_groups'][0]["lr"])
        for i, (inputs, info) in enumerate(trainDataLoader):
            #running_loss = 0
            #lam = 1 if epoch> 20 else 0
            lam = 1 if epoch> 20 else 1
            inputs = Variable(inputs).cuda().float()


            #heatmaps_targets = Variable(info["heatmaps"]).cuda()
            #mask,indices_valid = calculate_mask(heatmaps_targets)

            # celoss

            optimizer.zero_grad()
            class_output = net(inputs)

            loss_mask = info["loss_mask"].cuda()
            logits = class_output['pred_logits'].squeeze()
            #logits = torch.max(logits.reshape(-1, 6, 4), dim=2).values
            #labels = torch.max(info["label"].reshape(-1, 6, 4), dim=2).values
            #
            class_loss = class_criterion(logits, info["label"].float().cuda())
            #coord_loss = coord_criterion(class_output['pred_boxes'],info["keypoints"].float().cuda())
            coord_loss_no = coord_criterion_none(class_output['pred_boxes'], info["keypoints"].float().cuda())
            coord_loss = coord_loss_no.sum()
            #class_loss = class_loss.sum()
            #class_loss = torch.mean(torch.sum(class_loss, dim=(0,1)))

            total_loss = coord_loss+class_loss*lam
            total_loss.backward()
            optimizer.step()


            # 统计最大值与最小值
            if (i+1) % config['eval_freq'] == 0:
                print('---------------calculate loss-------')
                print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} ,CLASSLOSS:{},COORDSLOSS:{}'.format(
                    epoch, i * config['batch_size'],
                    sample_num, total_loss.item(),class_loss.item(),coord_loss.item()))



                # print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} class_loss:{:15} max : {:10} min : {}'.format(
                #     epoch, i * config['batch_size'],
                #     sample_num, running_loss.item(),class_loss.item(),v_max.item(),v_min.item()))
                # print('---------------calculate loss-------')
                # print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15}  max : {:10} min : {}'.format(
                #     epoch, i * config['batch_size'],
                #     sample_num, running_loss.item(),v_max.item(),v_min.item()))

            #评估



        train_loss.append(train_loss)

        lr_scheduler.step()

        print("using samples for testing",len(valDataset))
        with torch.no_grad():

            dict,summary,mean_auc,loss_mean =evaluate_Trans(model=net, dataloader=valDataLoader)
            #dict, summary, _,_ = evaluate_Trans(model=net, dataloader=TvalDataLoader)
            if mean_auc > best_auc:
                torch.save(net.module.state_dict() if len(gpus) > 1 else net.state_dict(),
                       config['checkout'] + 'SpineT_512_netvit_{seed}_best_model.ckpt'.format(seed=args.seed))
                best_auc = mean_auc
            print("best_auc is:", best_auc)
        vali_loss.append(loss_mean)
        #
        # torch.save(net.module.state_dict() if len(gpus) > 1 else net.state_dict(),
        #            config['checkout']+'kd_epoch_vit{epoch}_{seed}gray.ckpt'.format(epoch=epoch, seed=args.seed))
        # if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
        #     torch.save(net.module.state_dict()if len(gpus) > 1 else net.state_dict(),'./Checkpoints/kd_epoch_off{}_model.ckpt'.format(epoch))
            #evaluate_one(model=net, dataloader=valDataLoader)
    # plt.figure()
    plt.figure()
    plt.plot(train_loss, 'b-', label='train_loss')
    plt.plot(vali_loss, 'r-',label='val_loss')
    plt.ylabel('Train_loss')
    plt.xlabel('iter_num')
    plt.savefig(config['checkout']+'vit_loss{seed}.jpg'.format(seed=args.seed))

