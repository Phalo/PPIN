#coding=utf-8
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

# Five fold validations
from sklearn.model_selection import KFold

##Model Selection
from MMWUet import MMWUNet
from U2Netnew import U2Net
from Unet_Maw import UNet
from ED_optional import UResnet,BottleNeck,BasicBlock
from Unet_base import UNet_base
from torchvision.models.resnet import resnet50


from Unet_concate import UNet_conc
from Unet_GCN import UNet_GCN
from Y_Net import YNet


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


# torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别




def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,24,256,256)
    :return:numpy array (N,24,2) #
    """
    N,C,H,W = heatmaps.shape   # N= batch size C=24 hotmaps
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def get_mse(pred_points,gts,indices_valid=None):
    """

    :param pred_points: numpy (N,4,2)
    :param gts: numpy (N,4,2)
    :return:
    """
    pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    gts = gts[indices_valid[0],indices_valid[1],:]
    pred_points = Variable(torch.from_numpy(pred_points).float(),requires_grad=False)
    gts = Variable(torch.from_numpy(gts).float(),requires_grad=False)
    criterion = nn.MSELoss()
    loss = criterion(pred_points,gts)
    return loss

def calculate_mask(heatmaps_targets):
    """

    :param heatmaps_target: Variable (N,4,256,256)
    :return: Variable (N,4,256,256)
    """
    N,C,_,_ = heatmaps_targets.size()  #N =8 C = 4
    N_idx = []
    C_idx = []
    for n in range(N):      # 0-7
        for c in range(C):  # 0-3
            max_v = heatmaps_targets[n,c,:,:].max().item()
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = Variable(torch.zeros(heatmaps_targets.size()))
    mask[N_idx,C_idx,:,:] = 1.0
    mask = mask.float().cuda()
    return mask,[N_idx,C_idx]

def L1Loss(pred, gt, mask=None):
    # L1 Loss for offset map
    assert(pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        # Caculate grad of the area under mask
        distence = distence * mask
    return distence.sum() / mask.sum()

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])





if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--keypoint_model", type=str, default='baseline_lumbar',
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
    parser.add_argument("--attention", default=True, help="saptial attention")
    parser.add_argument("--keypoint_epochs", type=int, default=100,
                        help="max number of epochs, default: 100")
    parser.add_argument("--seed", type=int, default=96,
                        help="randomseed, default: 0")
    parser.add_argument("--keypoint_model_dir", type=str, default='./Checkpoints_final/',
                        help="saving keypoint model_dir")

    parser.add_argument('--keypoint_learning_rate', type=float, default=0.005,
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
    #config['lr_steps'] = [60, 80]
    config['lr_steps'] = [60, 80] # fir sa
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
                    classification_only=True,attention=True,bilinear=True)
            config['lr'] = 0.005

        else:
            net = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=True,attention=False)
            config['lr'] = 0.005

    elif args.keypoint_model == 'our+module1':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+ACL/'
        net = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=False)
        config['lr'] = 0.005

    elif args.keypoint_model == 'our+module2':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+WTS/'

        ## 0804add
        net = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=True)
        config['lr'] = 0.005

    elif args.keypoint_model == 'our+module1+2':
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/MAW/'
        net = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
                    classification_only=False)
        config['lr'] = 0.005

    elif args.keypoint_model == 'our+module1+3':
        net = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=True,
                    classification_only=False,bilinear=False)
        config['lr'] = 0.005
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+ACL+GCN/'
    elif args.keypoint_model== 'our_all':
        net = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=True,
                    classification_only=False)
        config['lr'] = 0.005
        #config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/ALL/'
    elif args.keypoint_model== 'Unet_our':
        net = UNet(in_channels=1, out_channels=24, classification=True )
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/cUnet_our/{}'.format(
            args.seed)
        config['lr'] = 0.005
    else:
        net = UNet_base(in_channels=1, out_channels=24, bilinear=True, classification=True)
        config['lr'] = 0.005
        config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/2019MICCAI/{}'.format(
        args.seed)
    print('the number of trainable parameters: %d' % get_number_of_learnable_parameters(net))
    print('Initial learning rate :',config['lr'])
    config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/Train_unify/{}'.format(args.seed)
    print ('Saving checkpoint to',config['checkout'])

    config['val'] ='right'



    gpus = [g for g in range(torch.cuda.device_count())]
    print(len(gpus))
    if len(gpus) > 1:
        net = nn.DataParallel(net, device_ids=gpus)
    net = net.cuda()

    criterion = nn.MSELoss(reduction='none')
    criterion = KpLoss()
    loss_regression_fn = L1Loss
    class_criterion = CLALoss()

    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_steps'], gamma=config['lr_gamma'])
    scaler = torch.cuda.amp.GradScaler() if config['amp'] else None


    # 定义 Dataset

    data_transforms = {
        "train": transform_new.Compose([

                                     # transform_new.Resize(H=224,W=224),

                                     transform_new.ToTensor(),
                                     transform_new.RandomHorizontalFlip(0.5),
                                     transform_new.Blur(),



                                     # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ]),

        "val" : transform_new.Compose([
            # transform_new.Resize(H=224, W=224),
            # transform_new.RandomHorizontalFlip(0),
            transform_new.ToTensor(),
                                    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])
    }
    trainDataset = KFDataset(config , mode='train', transforms=data_transforms["train"],fold= None,lumbar=False)
    # 定义 data loader
    trainDataLoader = DataLoader(trainDataset,config['batch_size'],True,num_workers=8)


    sample_num = len(trainDataset)
    print(sample_num)

    valDataset = KFDataset(config , mode='test', transforms=data_transforms["val"],fold= None,lumbar=False)
    valDataLoader = DataLoader(valDataset, 1, False, num_workers=8)




    if config['load_pretrained_weights']:
         if (config['checkout'] != ''):
             print("load dict from checkpoint")
             net.load_state_dict(torch.load(config['checkout']))
    train_loss = []
    val_loss = []
    best_auc = 0
    best_recall = 0
    best_epoch = 0
    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):
        net.float().cuda()
        net.train()
        print("current learn rate:", optimizer.state_dict()['param_groups'][0]["lr"])
        for i, (inputs, info) in enumerate(trainDataLoader):
            #lam = 10 if epoch> 20 else 1
            #lam = 0.05 * epoch if epoch > 20 else 1  #baseline
            lam = 0.05 * epoch if epoch >20 else 1  #our parameter 230804really
            inputs = Variable(inputs).cuda().float()


            heatmaps_targets = Variable(info["heatmaps"]).cuda()

            optimizer.zero_grad()
            outputs,regression_x,regression_y,class_output = net(inputs)
            outputs = outputs.to(torch.float32)
            loss_mask = info["loss_mask"].cuda()
            heatmaps_targets = heatmaps_targets.to(torch.float32)


            # final version
            labels = torch.max(info["label"].reshape(-1, 6, 4), dim=2).values

            #labels = info["label"].cuda()
            regression_loss_y = loss_regression_fn(regression_y, info["offset_y"].cuda(), heatmaps_targets)
            regression_loss_x = loss_regression_fn(regression_x, info["offset_x"].cuda(), heatmaps_targets)
            kp_loss = criterion(outputs, heatmaps_targets,loss_mask)
            # final version
            class_loss = class_criterion(class_output, labels ,loss_mask)
            running_loss = kp_loss +regression_loss_y + regression_loss_x + lam *class_loss
            running_loss.backward()
            optimizer.step()


            # 统计最大值与最小值
            v_max = torch.max(outputs)
            v_min = torch.min(outputs)
            if (i+1) % config['eval_freq'] == 0:
                print('---------------calculate loss-------')
                print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} CLASSLOSS:{} max : {:10} min : {}'.format(
                    epoch, i * config['batch_size'],
                    sample_num, running_loss.item(),class_loss.item(),v_max.item(),v_min.item()))
                # print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} class_loss:{:15} max : {:10} min : {}'.format(
                #     epoch, i * config['batch_size'],
                #     sample_num, running_loss.item(),class_loss.item(),v_max.item(),v_min.item()))
                # print('---------------calculate loss-------')
                # print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15}  max : {:10} min : {}'.format(
                #     epoch, i * config['batch_size'],
                #     sample_num, running_loss.item(),v_max.item(),v_min.item()))

            #评估

        lr_scheduler.step()
        train_loss.append(running_loss)
        print(len(valDataset))
        with torch.no_grad():
            dic, summary, mean_auc, mean_recall=evaluate_one(model=net, dataloader=valDataLoader)

            ##eval with 24classes
            #dic, summary, mean_auc, mean_recall = evaluate_all(model=net, dataloader=valDataLoader)
            if mean_auc > best_auc :
                torch.save(net.module.state_dict() if len(gpus) > 1 else net.state_dict(),
                           config['checkout'] + 'kd_net{model}_{seed}__{lr}best_{val}model.ckpt'.format(model=args.keypoint_model,seed=args.seed, lr=config['lr'],
                                                                                                        val=config['val']))
                best_auc = mean_auc
                best_epoch = epoch
            print("best_auc is:",best_auc )
            print("best_epoch is:",best_epoch)
                #best_recall = mean_recall
        if args.attention:
            torch.save(net.module.state_dict() if len(gpus) > 1 else net.state_dict(),
                   config['checkout']+'epoch_{model}{epoch}_{lr}_{val}_{seed}.ckpt'.format(model=args.keypoint_model,epoch=epoch, lr=config['lr'], val=config['val'],seed=args.seed))
        else:
            torch.save(net.module.state_dict() if len(gpus) > 1 else net.state_dict(),
                   config['checkout']+'epoch_{model}{epoch}__{lr}_{val}_{seed}.ckpt'.format(model=args.keypoint_model,epoch=epoch, lr=config['lr'], val=config['val'],seed=args.seed))
        # test_net = MMWUNet(in_channels=1,out_channels=24,classification=True).float().cuda()
        # test_net.load_state_dict(torch.load('./Checkpoints/Maw_at3/kd_epoch_near{}_model.ckpt'.format(epoch)))
        # with torch.no_grad():
        #     print("eval twice")
        #     evaluate_one(model=test_net, dataloader=valDataLoader)
        # if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
        #     torch.save(net.module.state_dict()if len(gpus) > 1 else net.state_dict(),'./Checkpoints/kd_epoch_off{}_model.ckpt'.format(epoch))
            #evaluate_one(model=net, dataloader=valDataLoader)
    plt.figure()
    plt.plot(train_loss, 'b-', label='Recon_loss')
    plt.ylabel('Train_loss')
    plt.xlabel('iter_num')
    plt.savefig(config['checkout']+'loss{SEED}_{lr}_{val}.jpg'.format(SEED=args.seed, lr=config['lr'],val=config['val']))


