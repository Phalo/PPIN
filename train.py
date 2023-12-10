#coding=utf-8
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint
import transforms
from data_loader import KFDataset
#from models import KFSGNet
import os
import argparse
#from multi_train_utils.distributed_utils import init_distributed_mode, dist ,cleanup ,reduce_value
#from train_eval import evaluate_one
#rom network import UNet_Pretrained
#from U2Net import U2Net
import matplotlib.pyplot as plt
from loss import KpLoss,CLALoss
import tempfile
config = dict()
config['lr'] = 0.01
config['momentum'] = 0.009
config['weight_decay'] = 1e-4
config['epoch_num'] = 100
config['batch_size'] = 2
config['sigma'] = 2.5
config['debug_vis'] = False

config['train_fname'] = ''
config['test_fname'] =''
#config ['path_image'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'
config ['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_test/'
config ['train_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_train/'

# config ['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/lumbar_test/'
# config ['train_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/lumbar_train/'

config['path_label'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/'
#config['path_label_train'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/train_json/'
config['path_label_train'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/lumbar_json'
#config['json_path']='/public/huangjunzhang/test/keypoints_train.json'
config['is_test'] = False

config['save_freq'] = 10
config['checkout'] = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints/kd_MLT_epoch_499_model.ckpt'
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
            nan = float('nan')
            #print(yy)
            y = yy[0] if yy[0]!= None else 0
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




#
# def init_distributed_mode(args):
#     #
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLDZ_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gps = args.rank % torch.cuda.device_count()
#
#     else:
#         print('Not using distributed mode')
#         args.distributed =False
#         return
#     args.distributed = True
#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     dist.barrier()


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    #init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    #weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

    #train_info, val_info, num_classes = read_split_data(args.data_path)
    #train_images_path, train_images_label = train_info
    #val_images_path, val_images_label = val_info
    pprint.pprint(config)


    #criterion2 = nn.P()
    #optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'] , weight_decay=config['weight_decay'])
    #optimizer = optim.Adam(net.parameters(),lr=config['lr'], weight_decay=config['weight_decay'])
    # optimizer = optim.RMSprop(net.parameters(),lr=config['lr'],
    #                                 weight_decay=config['weight_decay'],
    #                                 momentum=config['momentum'])
    # 定义 Dataset

    data_transforms = {
        "train": transforms.Compose([
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.Blur(),
                                     transforms.Brightness(),
                                     transforms.ToTensor(),

                                     # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                     ]),

        "val" : transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])
    }


    trainDataset = KFDataset(config , mode='train', transforms=data_transforms["train"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
    testDataset = KFDataset(config, mode='test',transforms=data_transforms["val"])
    test_sampler = torch.utils.data.distributed.DistributedSampler(testDataset)
    #testDataLoader = DataLoader(testDataset,1, True, num_workers=8)
    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    #trainDataset.load()
    # 定义 data loader
    sample_num = len(trainDataset)
    print(sample_num)



    trainDatasetloader = torch.utils.data.DataLoader(trainDataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)
                                               #collate_fn=trainDataset.collate_fn)
    testDataLoader = torch.utils.data.DataLoader(testDataset,
                                               batch_sampler=test_sampler,
                                               pin_memory=True,
                                               num_workers=nw)
                                               #collate_fn=trainDataset.collate_fn)



    # collate_fn=trainDataset.collate_fn)

    torch.manual_seed(0)
    cudnn.benchmark = True
    #model = KFSGNet()
    model = U2Net(in_channels=1,out_channels=24)
    model.float().cuda().to(device)
    if config['load_pretrained_weights']:
         if (config['checkout'] != ''):
             print("load dict from checkpoint")
             model.load_state_dict(torch.load(config['checkout']))

    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(),"initial_weights.pt")
        if rank ==0 :
            torch.save(model.state_dict(),checkpoint_path)
        dist.barrier()

        model.load_state_dict(torch.load(checkpoint_path,map_location=device))
    # convert to DDP model
    if args.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    #print('test')

    #criterion = nn.MSELoss(reduction='sum')
    #criterion = nn.BCELoss()
    criterion = KpLoss()
    class_criterion = CLALoss()
    train_loss = []
    val_loss = []
    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):
        train_sampler.set_epoch(epoch)
        model.train()
        for i, (inputs, heatmaps_targets, gts, loss_mask, label ) in enumerate(trainDatasetloader):

            lam = 0.01
            inputs = Variable(inputs).cuda().float()

            heatmaps_targets = Variable(heatmaps_targets).cuda()
            mask, indices_valid = calculate_mask(heatmaps_targets)

            optimizer.zero_grad()
            outputs, class_output = model(inputs)
            outputs = outputs.to(torch.float32)

            heatmaps_targets = heatmaps_targets.to(torch.float32)
            # print(torch.max(outputs[0]), torch.min(outputs[0]))
            # print(torch.max(heatmaps_targets[0]),torch.min(heatmaps_targets[0]))
            outputs = outputs * mask
            heatmaps_targets = heatmaps_targets * mask

            kp_loss = criterion(outputs, heatmaps_targets, loss_mask)
            class_loss = class_criterion(class_output, label, loss_mask)
            running_loss = kp_loss + class_loss
            running_loss.backward()
            running_loss = reduce_value(running_loss, average=True)
            optimizer.step()

            # 统计最大值与最小值
            v_max = torch.max(outputs)
            v_min = torch.min(outputs)
            if (i + 1) % config['eval_freq'] == 0:
                print('---------------calculate loss-------')
                print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:15} CLASSLOSS:{} max : {:10} min : {}'.format(
                    epoch, i * config['batch_size'],
                    sample_num, running_loss.item(), class_loss.item(), v_max.item(), v_min.item()))

        train_loss.append(running_loss)

        if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            torch.save(model.state_dict(),'./Checkpoints/kd_MLTGPU_epoch_{}_model.ckpt'.format(epoch))
            #evaluate_one(model=model, data_loader=valdatasetloader)
            #evaluate_one(model=model, dataloader=testDataLoader)

    plt.figure()
    plt.plot(train_loss, 'b-', label='Recon_loss')
    plt.ylabel('Train_loss')
    plt.xlabel('iter_num')
    plt.show()

if __name__ == '__main__':




    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config['epoch_num'])
    parser.add_argument('--batch-size', type=int, default=config['batch_size'])
    parser.add_argument('--lr', type=float, default=config['lr'])
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)



    # parser.add_argument('--weights', type=str, default='resNet34.pth',
    #                     help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)
