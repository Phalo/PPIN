# coding=utf-8

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import KFDataset

import time
#from train_single_gpu import config, get_peak_points, get_mse
#
from collections.abc import Iterable
from sklearn import metrics

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

# config ['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/Rsize/'
# config ['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/Lsize/'

# config ['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/lumbar_test/'
# config ['train_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/lumbar_train/'

config['path_label'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/'
config['path_label_train'] = '/public/huangjunzhang/KeyPointsDetection-master/txt/train_json/'
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


from MMWUet import MMWUNet
from U2Netnew import U2Net
from Unet_Maw import UNet
#from Unet_base import UNet_base



from Unet_concate import UNet_conc
from Unet_GCN import UNet_GCN
from Y_Net import YNet
from ED_optional import UResnet,BottleNeck,BasicBlock

# THRESHOLD = [1,1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
THRESHOLD =[3,6,9,10]
DRAW_TEXT_SIZE_FACTOR = { 'cephalometric': 1.13, 'hand': 1, 'chest': 1.39}

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


def np2py(obj):
    if isinstance(obj, Iterable):
        return [np2py(i) for i in obj]
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj
 #计算距离
def radial(pt1, pt2 ,factor=1):
    if  not isinstance(factor,Iterable):
        factor = [factor]*len(pt1)
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5



def get_sdr(distance_list, threshold =THRESHOLD):
    ret = {}
    n = len(distance_list)
    for th in threshold:
        ret[th]=sum(d<= th for d in distance_list)/n
    return ret

def cal_all_distance(points, gt_points, factor ):
    n1 = len(points)
    n2 = len(gt_points)
    factor_index = np.array(factor) * 1
    if n1 == 0:
        print("[Warning]: Empty input for calculating mean and std")
        return 0, 0
    if n1 != n2:
        raise Exception("Error: lengthes dismatch, {}<>{}".format(n1, n2))
    return [radial(p, q, factor) for p, q ,factor in zip(points, gt_points,factor_index)]

def analysis_all(li1):
    summary = {}
    mean1, std1, = np.mean(li1), np.std(li1)
    sdr1 = get_sdr(li1)
    n = len(li1)
    summary['LANDMARK_NUM'] = n
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    print('MRE:', mean1)
    print('STD:', std1)
    print('SDR:')
    for k in sorted(sdr1.keys()):
        print('     {}: {}'.format(k, sdr1[k]))
    return summary
def analysis(li1):
    summary = {}
    mean1, std1, = np.mean(li1), np.std(li1)
    sdr1 = get_sdr(li1)
    n = len(li1)
    summary['LANDMARK_NUM'] = n
    summary['MRE'] = np2py(mean1)
    summary['STD'] = np2py(std1)
    summary['SDR'] = {k: np2py(v) for k, v in sdr1.items()}
    # print('MRE:', mean1)
    # print('STD:', std1)
    # print('SDR:')
    # for k in sorted(sdr1.keys()):
    #     print('     {}: {}'.format(k, sdr1[k]))
    return summary

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    print(optimal_threshold)
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def auc_curve(index_name,y,prob):
    sns.set(font_scale=1.2)
    plt.rc('font', family='Times New Roman')
    fpr, tpr, thresholds = metrics.roc_curve(y,prob)
    roc_auc = metrics.auc(fpr,tpr) ###计算auc的值
    lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    print(optimal_point)
    # plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
    # plt.text(optimal_point[0], optimal_point[1], (float('%.2f'% optimal_point[0]),
    #                                               float('%.2f'% optimal_point[1])),
    #                                                 ha='right', va='top', fontsize=12)
    # #plt.text(optimal_point[0], optimal_point[1],  f'Threshold:{optimal_th:.2f}', fontsize=12)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate',fontsize = 14)
    # plt.ylabel('True Positive Rate',fontsize = 14)
    # #plt.title('ROC analysis of '+ index_name,fontsize = 14)
    # plt.legend(loc="lower right",fontsize = 12)
    # plt.show()

    return optimal_th

def evaluate_fft(model, dataloader,lumbar=False):
    # 加载模型
    model.eval()
    # if (config['checkout'] != ''):
    #    net.load_state_dict(torch.load(config['checkout']))
    index = [
        'C2_TR',
        'C2_TL',
        'C2_DR',
        'C2_DL',
        'C3_TR',
        'C3_TL',
        'C3_DR',
        'C3_DL',
        'C4_TR',
        'C4_TL',
        'C4_DR',
        'C4_DL',
        'C5_TR',
        'C5_TL',
        'C5_DR',
        'C5_DL',
        'C6_TR',
        'C6_TL',
        'C6_DR',
        'C6_DL',
        'C7_TR',
        'C7_TL',
        'C7_DR',
        'C7_DL']
    dic = {}
    summary = {}
    aucs = []
    accs = []
    recalls = []
    f1s=[]
    sensitivities = []
    specificities = []
    distance_list = []
    gt_point_group = []
    class_label_group = []

        #class_label_all = torch.zeros(1,24)
    for i, (images, info) in enumerate(dataloader):
        images = Variable(images).float().cuda()
        gt = Variable(info["keypoints"]).float().cuda()
        gt_point = gt.cpu().data.numpy().reshape((-1, 2))
        #label = torch.as_tensor(info["label"], dtype=int).tolist()[0]
        label = torch.as_tensor(info["label"], dtype=int)
        pred_heatmaps, regress_x, regress_y = model.forward(images)







        #plot
        demo_img = images[0].cpu().data.numpy().reshape((2,512, 512))
        demo_img = (demo_img * 255.).astype(np.uint8)
        # plot output
        demo_heatmaps_output = pred_heatmaps[0].cpu().data.numpy()  # 24*512*512

        demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()[np.newaxis, ...]
        demo_pred_poins = get_peak_points(demo_heatmaps)[0]
        # plt.figure(2)
        # plt.imshow(demo_img, cmap=plt.get_cmap('gray'))
        # for i in range(24):
        #     plt.subplot(4, 6, i + 1)
        # #     #plt.scatter(gt_point[i][0], gt_point[i][1])
        # #     #plt.text(gt_point[i][0], gt_point[i][1], '{}'.format(index[i]), color='g')
        # #     plt.imshow(demo_img,cmap=plt.get_cmap('gray'))
        #     plt.imshow(demo_heatmaps_output[i], cmap=plt.get_cmap('gray'), alpha=.5)
        #     plt.scatter(gt_point[i][0], gt_point[i][1], color='g',s=5)
        #     #plt.scatter(demo_pred_poins[i][0], demo_pred_poins[i][1], color='r', s=5)
        # #plt.show()
        # #for i in range(len(gt_point)):
        #
        #
        #     #plt.imshow(demo_heatmaps_output[i], cmap=plt.get_cmap('gray'), alpha=.5)
        #     #plt.text(gt_point[i][0], gt_point[i][1], '{}'.format(index[i]), color='g')
        # plt.show()
        # demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()



        #计算距离
        cur_distance_list = cal_all_distance(gt_point,demo_pred_poins , info["loss_mask"].squeeze())
        x = 0
        length=len(cur_distance_list)
        while x< length:
            if cur_distance_list[x]==0 :
                del cur_distance_list[x]
                x-=1
                length-= 1
            x+=1

        distance_list += cur_distance_list
        dic[i] = distance_list
        # 评价指标

    ##divided by group

    li_total = []
    for d, cur_distance_list in dic.items():
        summary[d] = analysis(cur_distance_list)
        li_total += cur_distance_list
    summary['total'] = analysis_all(li_total)




        #plt.show()

    return dic, summary

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def evaluate_one(model, dataloader,lumbar=False):
    # 加载模型
    model.eval()
    # if (config['checkout'] != ''):
    #    net.load_state_dict(torch.load(config['checkout']))
    index = [
        '2TR',
        '2TL',
        '2DR',
        '2DL',
        '3TR',
        '3TL',
        '3DR',
        '3DL',
        '4DR',
        '4TL',
        '4DR',
        '4DL',
        '5TR',
        '5TL',
        '5DR',
        '5DL',
        '6TR',
        '6TL',
        '6DR',
        '6DL',
        '7TR',
        '7TL',
        '7DR',
        '7DL']
    dic = {}
    summary = {}
    aucs = []
    accs = []
    recalls = []
    f1s=[]
    sensitivities = []
    specificities = []
    distance_list = []
    gt_point_group = []
    class_label_group = []

        #class_label_all = torch.zeros(1,24)
    for i, (images, info) in enumerate(dataloader):
        images = Variable(images).float().cuda()
        gt = Variable(info["keypoints"]).float().cuda()
        gt_point = gt.cpu().data.numpy().reshape((-1, 2))
        #label = torch.as_tensor(info["label"], dtype=int).tolist()[0]
        label = torch.as_tensor(info["label"], dtype=int)
        pred_heatmaps, regress_x, regress_y,class_label = model.forward(images)
        # print(pred_heatmaps.cpu().data.numpy())
        #class_label = [bs,24]



        #tensor2list
        # class_label = torch.detach(class_label)
        # class_label = class_label.tolist()[0]
        # label = torch.detach(label)
        # label = label.tolist()[0]
        # gt_point_group = []
        # class_label_group = []
        # #divided in groups
        # for j in range(0,len(class_label),4):
        #
        #     if sum(label[j:j+4])==0:
        #         gt_point_group.append(0)
        #     else:
        #         gt_point_group.append(1)
        #     class_label_group.append(max(class_label[j:j+4]))
        #
        #
        #
        #
        # class_label_group = torch.as_tensor(class_label_group, dtype=int)
        # gt_point_group = torch.as_tensor(gt_point_group, dtype=int)


        if i==0:

            # pred_label_all = class_label_group
            # label_gt_all = gt_point_group

            pred_label_all = class_label
            label_gt_all = label
        else:
            pred_label_all = torch.cat([pred_label_all, class_label])
            label_gt_all = torch.cat([label_gt_all, label])

            # pred_label_all = torch.cat([pred_label_all, class_label_group])
            # label_gt_all = torch.cat([label_gt_all, gt_point_group])



        #plot
        demo_img = images[0].cpu().data.numpy().reshape((-1,512, 512))
        demo_img = (demo_img * 255.).astype(np.uint8)
        # plot output
        demo_heatmaps_output = pred_heatmaps[0].max(0)[0].cpu().data.numpy()  # 24*512*512

        demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()[np.newaxis, ...]
        demo_pred_poins = get_peak_points(demo_heatmaps)[0]

        # plt.figure(2)
        # plt.imshow(demo_img[0], cmap=plt.get_cmap('gray'))
        # for i in range(24):
        #     #plt.subplot(4, 6, i + 1)
        # #     #plt.scatter(gt_point[i][0], gt_point[i][1])
        #
        # #     plt.imshow(demo_img,cmap=plt.get_cmap('gray'))
        #     plt.imshow(demo_heatmaps_output, cmap=plt.get_cmap('jet'), alpha=.5)
        #     plt.scatter(gt_point[i][0], gt_point[i][1], color='black',s=5)
        #     plt.text(gt_point[i][0], gt_point[i][1], '{}'.format(index[i]), color='g',fontsize=5)
        #     plt.scatter(demo_pred_poins[i][0], demo_pred_poins[i][1], color='w', s=5)
        #     plt.text(demo_pred_poins[i][0], demo_pred_poins[i][1], '{}'.format(index[i]), color='r')
        # plt.savefig("./SR/{}gt.jpg".format(info["image_id"]))
        # print("saved down")
        # plt.close()



        #plt.show()
        # #for i in range(len(gt_point)):
        #
        #
        #     #plt.imshow(demo_heatmaps_output[i], cmap=plt.get_cmap('gray'), alpha=.5)
        #     #plt.text(gt_point[i][0], gt_point[i][1], '{}'.format(index[i]), color='g')
        # plt.show()
        # demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()








        #print('pred',demo_pred_poins)
        #print('gt',gt)
        #calculate acc

        #print(acc)


        #计算距离
        cur_distance_list = cal_all_distance(gt_point,demo_pred_poins , info["loss_mask"].squeeze())
        x = 0
        length=len(cur_distance_list)
        while x< length:
            if cur_distance_list[x]==0 :
                del cur_distance_list[x]
                x-=1
                length-= 1
            x+=1

        distance_list += cur_distance_list
        dic[i] = distance_list
        # 评价指标

    ##divided by group
    if pred_label_all.size(1)==24:

        pred_label_all = torch.max(pred_label_all.reshape(-1,6,4),dim=2).values

    if lumbar:
        label_gt_all = torch.max(label_gt_all.reshape(-1,5,4),dim=2).values
    else:
        label_gt_all = torch.max(label_gt_all.reshape(-1, 6, 4), dim=2).values

    for lab in range(pred_label_all.size(1)):
        #
        pred_bool = []
        #pred_label= pred_label_all[:, lab].tolist()
        pred_label = pred_label_all[:, lab].tolist()

        #label_gt = label_gt_all[:, lab].tolist()
        label_gt = label_gt_all[:, lab].tolist()
        # print(label_gt)

        # for pred_lab in pred_label:
        #     if pred_lab >0.2:
        #         pred_bool.append(1)
        #     else:
        #         pred_bool.append(0)
        #print(sum(label_gt))
        if sum(label_gt) != 0 :
            auc = metrics.roc_auc_score(label_gt, pred_label)
            thresholds= auc_curve(index_name=lab, y=label_gt, prob=pred_label)

            # if thresholds >= 1 :
            #     thresholds= 0.5
            print('thresholds:', thresholds)
            for pred_lab in pred_label:
                if pred_lab > thresholds:
                    pred_bool.append(1)
                else:
                    pred_bool.append(0)
            acc = metrics.balanced_accuracy_score(label_gt, pred_bool)
            tn, fp, fn, tp = metrics.confusion_matrix(label_gt, pred_bool).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            recall = metrics.recall_score(label_gt,pred_bool)
            f1 = metrics.f1_score(label_gt,pred_bool)
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)
            accs.append(acc)
    mean_f1 = np.around(np.mean(np.array(f1s)), decimals=4)
    mean_recall = np.around(np.mean(np.array(recalls)), decimals=4)
    mean_sensitivity = np.around(np.mean(np.array(sensitivities)),decimals=4)
    mean_specificity = np.around(np.mean(np.array(specificities)),decimals=4)
    mean_auc = np.around(np.mean(np.array(aucs)), decimals=4)
    mean_acc = np.around(np.mean(np.array(accs)), decimals=4)



    print("aucs:", aucs)
    print("accs:", accs)
    print("recalls", recalls)
    print("specificity",specificities)
    print("f1", f1s)

    print("mean_aucs:", mean_auc)
    print("mean_accs", mean_acc)
    print("mean_recall", mean_recall)
    print("mean_specificity",mean_specificity)
    print("mean_f1", mean_f1)
    li_total = []
    for d, cur_distance_list in dic.items():
        summary[d] = analysis(cur_distance_list)
        li_total += cur_distance_list
    summary['total'] = analysis_all(li_total)




        #plt.show()

    return dic, summary, mean_auc, mean_recall

if __name__ == '__main__':
    from dataloader_new import KFDataset
    import transform_new
    import glob
    from Unet_base import UNet_base
    import os
    from sklearn.model_selection import KFold

    config['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/dataloader_test/'
    #config['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/Rsize/'
    #config['test_image_path'] = '/public/huangjunzhang/KeyPointsDetection-master/Lsize/'
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
    # df = pd.DataFrame(data=train_files, index=['0', '1', '2', '3',
    data_transforms = {
        "train": transform_new.Compose([transform_new.ToTensor(),
                                        #transform_new.RandomHorizontalFlip(1),
                                     # transform_new.RandomHorizontalFlip(1),
                                        ]),
        "val" : transform_new.Compose([transform_new.ToTensor()])
    }





    model = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=True,
                  classification_only=False).cuda()
    #model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/ALL/kd_net_best_model.ckpt"
    model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/1kd_net_1_bestAG_model.ckpt"

    #model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/Train_R/68kd_netour_all_68__0.005bestconv_model.ckpt"
    # #model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/flip/98kd_netour_all_98__0.005bestconv_model.ckpt"
    # model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/flip/68kd_netour_all_68__0.005bestconv_model.ckpt"
    model.load_state_dict(torch.load(model_dir))

    # model = UResnet(block=BottleNeck, layers=[3, 4, 6, 3], num_classes=24,attention=False).cuda()
    # model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/Mlt_highest_sap/kd_net_0.001best_model.ckpt"
    # model.load_state_dict(torch.load(model_dir))

    # headlocnet
    # model = UNet_conc(in_channels=1, out_channels=24, bilinear=True, classification=True, attention=True).cuda()
    # model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/2020Wang/kd_net_256_best_model.ckpt"
    # model.load_state_dict(torch.load(model_dir))

    ###Ynet
    # model = YNet(in_channels=1, out_channels=24, bilinear=True, classification=True).cuda()
    # model_dir = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/YNet/kd_net_0.0001sap_best_model.ckpt'
    # model.load_state_dict(torch.load(model_dir))


    #AFEN
    # model = UNet_GCN(in_channels=1, out_channels=24, bilinear=True, classification=True).cuda()
    # model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/2020shen/kd_net_128_best_model.ckpt"
    # model.load_state_dict(torch.load(model_dir))

    # # cUnet
    # model = UNet_base(in_channels=1,out_channels=24,classification=False).cuda()
    # model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/2019MICCAI/epoch_cUnet98_sap_128.ckpt"
    # model.load_state_dict(torch.load(model_dir))



    # model = U2Net(in_channels=1, out_channels=24, classification=True, classification_gcn=False,classification_only=True,attention=True).cuda()
    # model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/baseline/epoch_baseline88_0.005_AG128.ckpt"
    # model.load_state_dict(torch.load(model_dir))


    ## 0804add
    # model = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
    #               classification_only=True).cuda()
    # model_dir = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+WTS+wos/kd_epoch_MAWnet36_gcn_LAST.ckpt'
    #print(model)

    # model = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
    #               classification_only=False).cuda()
    #model_dir = '/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/U2+ACL/kd_net_128_best_model.ckpt'
    #
    ##3 module 1+2
    #config['checkout'] =
    # model = MMWUNet(in_channels=1, out_channels=24, classification=True, classification_gcn=False,
    #               classification_only=False).cuda()
    # model_dir ='/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_final/ablation/MAW/kd_net_256_best_model.ckpt'
    # model.load_state_dict(torch.load(model_dir))

    for epoch in range(0,1):
        print('eval epoch{}'.format(epoch))
        print('validation from test_files',epoch)
        valDataset = KFDataset(config, mode='test', transforms=data_transforms["train"], fold=None)
        valDataLoader = DataLoader(valDataset, 1, False, num_workers=0)
        print(len(valDataset))
        # model_dir = "/public/huangjunzhang/KeyPointsDetection-master/Checkpoints_2308/SA+WTS+BAM/98kd_netour+module2_98__0.01bestconv_model.ckpt"
        #print('loading from model_dir:',model_dir)

        #model.load_state_dict(torch.load('./Checkpoints_final/baseline/kd_epoch_baseline{}_0.005.ckpt'.format(epoch)), strict=False)
        #model.load_state_dict(torch.load('./Checkpoints_final/Mlt_highest_sap/kd_epoch_Mlt_highest_sap{}_0.001.ckpt'.format(epoch)),strict=False)
        #model.load_state_dict(torch.load('./Checkpoints_final/2020Wang/kd_epoch_2020Wang{}_0.0005.ckpt'.format(epoch)),strict=False)
        #model.load_state_dict(torch.load('./Checkpoints_final/Mlt_lowest_sap/kd_epoch_U2net{}_low_sap_model.ckpt'.format(epoch)),strict=False)

        with torch.no_grad():
            t_start = time_synchronized()
            evaluate_one(model=model, dataloader=valDataLoader)
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

    # test_net = MMWUNet(in_channels=1, out_channels=24, classification=True).float().cuda()
    # test_net.load_state_dict(torch.load('./Checkpoints/Maw/kd_epoch_Harr39_model.ckpt'))
    # with torch.no_grad():
    #     print("eval twice")
    #     evaluate_one(model=test_net, dataloader=valDataLoader)
